"""TDA-Based Portfolio Construction.

Phase 5: Constructs portfolios using topological features.
Uses cluster structure, centrality weights, and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from src.tda_stock_selector import StockScore, ClusterAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """A single position in the portfolio."""
    ticker: str
    weight: float           # Portfolio weight (0-1)
    cluster_id: int         # Cluster assignment
    entry_score: float      # Score at entry
    target_weight: float    # Target weight before adjustments


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation."""
    positions: List[PortfolioPosition]
    cash_weight: float
    total_exposure: float   # Sum of position weights
    n_positions: int
    n_clusters: int         # Cluster diversification
    concentration: float    # HHI concentration index
    expected_vol: float     # Expected portfolio volatility


class TDAPortfolioConstructor:
    """
    Construct portfolios using TDA-derived insights.
    
    Weighting Methods:
    1. Equal Weight: Simple 1/N allocation
    2. Score Weight: Weight by TDA score
    3. Inverse Vol: Weight by 1/volatility
    4. Risk Parity: Equal risk contribution
    5. Cluster Balanced: Equal weight per cluster, then per stock
    
    Constraints:
    - Maximum position size
    - Maximum cluster exposure
    - Minimum diversification (n positions)
    - Total exposure limit (based on regime)
    """
    
    def __init__(
        self,
        max_position_weight: float = 0.10,   # 10% max per position
        max_cluster_weight: float = 0.30,    # 30% max per cluster
        min_position_weight: float = 0.01,   # 1% min per position
        min_positions: int = 10,             # Minimum positions for diversification
    ):
        """
        Initialize portfolio constructor.
        
        Args:
            max_position_weight: Maximum weight for any single position
            max_cluster_weight: Maximum weight for any single cluster
            min_position_weight: Minimum weight to include a position
            min_positions: Minimum number of positions
        """
        self.max_position_weight = max_position_weight
        self.max_cluster_weight = max_cluster_weight
        self.min_position_weight = min_position_weight
        self.min_positions = min_positions
    
    def construct_portfolio(
        self,
        selected_stocks: List[StockScore],
        cluster_analysis: Dict[int, ClusterAnalysis],
        returns_df: pd.DataFrame,
        regime: str = "BULL",
        method: str = "score_weight",
        target_exposure: float = None,
    ) -> PortfolioAllocation:
        """
        Construct portfolio from selected stocks.
        
        Args:
            selected_stocks: List of StockScore from selector
            cluster_analysis: Cluster analysis from selector
            returns_df: Returns DataFrame for volatility estimation
            regime: Current market regime
            method: Weighting method
            target_exposure: Target total exposure (None = derive from regime)
            
        Returns:
            PortfolioAllocation with positions and metadata
        """
        if len(selected_stocks) == 0:
            return self._empty_portfolio()
        
        # Determine target exposure from regime
        if target_exposure is None:
            target_exposure = self._regime_exposure(regime)
        
        # Get raw weights based on method
        if method == "equal_weight":
            weights = self._equal_weight(selected_stocks)
        elif method == "score_weight":
            weights = self._score_weight(selected_stocks)
        elif method == "inverse_vol":
            weights = self._inverse_vol_weight(selected_stocks, returns_df)
        elif method == "risk_parity":
            weights = self._risk_parity_weight(selected_stocks, returns_df)
        elif method == "cluster_balanced":
            weights = self._cluster_balanced_weight(selected_stocks)
        else:
            weights = self._equal_weight(selected_stocks)
        
        # Apply constraints
        weights = self._apply_position_constraints(weights, selected_stocks)
        weights = self._apply_cluster_constraints(weights, selected_stocks)
        
        # Normalize to target exposure
        weights = self._normalize_weights(weights, target_exposure)
        
        # Build positions
        positions = []
        for stock, weight in zip(selected_stocks, weights):
            if weight >= self.min_position_weight:
                positions.append(PortfolioPosition(
                    ticker=stock.ticker,
                    weight=weight,
                    cluster_id=stock.cluster_id,
                    entry_score=stock.total_score,
                    target_weight=weight / target_exposure if target_exposure > 0 else 0,
                ))
        
        # Calculate portfolio metrics
        total_exposure = sum(p.weight for p in positions)
        cash_weight = 1.0 - total_exposure
        
        # Concentration (HHI)
        hhi = sum(p.weight ** 2 for p in positions) if positions else 0
        
        # Cluster diversification
        n_clusters = len(set(p.cluster_id for p in positions))
        
        # Expected volatility
        exp_vol = self._estimate_portfolio_vol(positions, returns_df)
        
        return PortfolioAllocation(
            positions=positions,
            cash_weight=cash_weight,
            total_exposure=total_exposure,
            n_positions=len(positions),
            n_clusters=n_clusters,
            concentration=hhi,
            expected_vol=exp_vol,
        )
    
    def _empty_portfolio(self) -> PortfolioAllocation:
        """Return empty portfolio (100% cash)."""
        return PortfolioAllocation(
            positions=[],
            cash_weight=1.0,
            total_exposure=0.0,
            n_positions=0,
            n_clusters=0,
            concentration=0.0,
            expected_vol=0.0,
        )
    
    def _regime_exposure(self, regime: str) -> float:
        """Get target exposure based on regime."""
        exposures = {
            "BULL": 1.0,
            "RECOVERY": 0.8,
            "TRANSITION": 0.5,
            "BEAR": 0.3,
            "CRISIS": 0.1,
        }
        return exposures.get(regime, 0.6)
    
    def _equal_weight(self, stocks: List[StockScore]) -> np.ndarray:
        """Equal weight allocation."""
        n = len(stocks)
        return np.ones(n) / n
    
    def _score_weight(self, stocks: List[StockScore]) -> np.ndarray:
        """Weight by TDA score."""
        scores = np.array([max(s.total_score, 0.1) for s in stocks])
        return scores / scores.sum()
    
    def _inverse_vol_weight(
        self,
        stocks: List[StockScore],
        returns_df: pd.DataFrame,
    ) -> np.ndarray:
        """Weight by inverse volatility."""
        weights = []
        for stock in stocks:
            vol = stock.volatility
            if vol <= 0 or np.isnan(vol):
                vol = 0.3  # Default
            weights.append(1.0 / vol)
        
        weights = np.array(weights)
        return weights / weights.sum()
    
    def _risk_parity_weight(
        self,
        stocks: List[StockScore],
        returns_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Risk parity weighting (simplified).
        Each position contributes equal risk.
        """
        # Get volatilities
        vols = np.array([max(s.volatility, 0.1) for s in stocks])
        
        # Inverse vol squared (approximation)
        weights = 1.0 / (vols ** 2)
        
        return weights / weights.sum()
    
    def _cluster_balanced_weight(
        self,
        stocks: List[StockScore],
    ) -> np.ndarray:
        """
        Cluster-balanced weighting.
        Equal weight to each cluster, then equal weight within cluster.
        """
        # Group by cluster
        cluster_stocks = {}
        for i, stock in enumerate(stocks):
            cid = stock.cluster_id
            if cid not in cluster_stocks:
                cluster_stocks[cid] = []
            cluster_stocks[cid].append(i)
        
        n_clusters = len(cluster_stocks)
        weight_per_cluster = 1.0 / n_clusters
        
        weights = np.zeros(len(stocks))
        for cid, indices in cluster_stocks.items():
            weight_per_stock = weight_per_cluster / len(indices)
            for idx in indices:
                weights[idx] = weight_per_stock
        
        return weights
    
    def _apply_position_constraints(
        self,
        weights: np.ndarray,
        stocks: List[StockScore],
    ) -> np.ndarray:
        """Apply maximum position size constraint."""
        # Cap at max weight
        weights = np.clip(weights, 0, self.max_position_weight)
        
        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def _apply_cluster_constraints(
        self,
        weights: np.ndarray,
        stocks: List[StockScore],
    ) -> np.ndarray:
        """Apply maximum cluster exposure constraint."""
        # Calculate cluster weights
        cluster_weights = {}
        for i, stock in enumerate(stocks):
            cid = stock.cluster_id
            if cid not in cluster_weights:
                cluster_weights[cid] = 0
            cluster_weights[cid] += weights[i]
        
        # Scale down over-exposed clusters
        for cid, total_weight in cluster_weights.items():
            if total_weight > self.max_cluster_weight:
                scale = self.max_cluster_weight / total_weight
                for i, stock in enumerate(stocks):
                    if stock.cluster_id == cid:
                        weights[i] *= scale
        
        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def _normalize_weights(
        self,
        weights: np.ndarray,
        target_exposure: float,
    ) -> np.ndarray:
        """Normalize weights to target exposure."""
        if weights.sum() > 0:
            weights = weights * target_exposure / weights.sum()
        return weights
    
    def _estimate_portfolio_vol(
        self,
        positions: List[PortfolioPosition],
        returns_df: pd.DataFrame,
    ) -> float:
        """Estimate portfolio volatility."""
        if len(positions) == 0:
            return 0.0
        
        # Get returns for portfolio stocks
        tickers = [p.ticker for p in positions]
        weights = np.array([p.weight for p in positions])
        
        valid_tickers = [t for t in tickers if t in returns_df.columns]
        if len(valid_tickers) < 2:
            return 0.3  # Default
        
        # Get covariance matrix
        returns = returns_df[valid_tickers].iloc[-60:]
        cov = returns.cov() * 252  # Annualized
        
        # Adjust weights for missing tickers
        valid_weights = []
        for t, w in zip(tickers, weights):
            if t in valid_tickers:
                valid_weights.append(w)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum() if valid_weights.sum() > 0 else valid_weights
        
        # Portfolio variance
        port_var = valid_weights @ cov.values @ valid_weights
        
        return np.sqrt(port_var)
    
    def rebalance(
        self,
        current_positions: List[PortfolioPosition],
        new_allocation: PortfolioAllocation,
        threshold: float = 0.02,  # 2% drift threshold
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """
        Generate rebalancing trades.
        
        Args:
            current_positions: Current portfolio positions
            new_allocation: New target allocation
            threshold: Drift threshold to trigger trade
            
        Returns:
            Tuple of (sells, buys, weight_changes)
        """
        current_weights = {p.ticker: p.weight for p in current_positions}
        target_weights = {p.ticker: p.weight for p in new_allocation.positions}
        
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        sells = []
        buys = []
        changes = {}
        
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            
            change = target - current
            
            if abs(change) > threshold:
                if change < 0:
                    sells.append(ticker)
                else:
                    buys.append(ticker)
                changes[ticker] = change
        
        return sells, buys, changes


class TDAPortfolioBacktester:
    """
    Backtest TDA-based portfolio construction.
    """
    
    def __init__(
        self,
        constructor: TDAPortfolioConstructor = None,
        rebalance_frequency: int = 5,  # Days between rebalances
    ):
        """
        Initialize backtester.
        
        Args:
            constructor: Portfolio constructor
            rebalance_frequency: Days between rebalancing
        """
        self.constructor = constructor or TDAPortfolioConstructor()
        self.rebalance_frequency = rebalance_frequency
    
    def backtest(
        self,
        allocations: List[PortfolioAllocation],
        returns_df: pd.DataFrame,
        dates: List[str],
    ) -> pd.DataFrame:
        """
        Backtest portfolio allocations.
        
        Args:
            allocations: Daily portfolio allocations
            returns_df: Full returns DataFrame
            dates: Date labels
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        equity = 1.0
        
        for i, (date, allocation) in enumerate(zip(dates, allocations)):
            # Calculate daily return
            portfolio_return = 0.0
            
            for position in allocation.positions:
                if position.ticker in returns_df.columns:
                    stock_return = returns_df[position.ticker].iloc[i]
                    if not np.isnan(stock_return):
                        portfolio_return += position.weight * stock_return
            
            # Update equity
            equity *= (1 + portfolio_return)
            
            results.append({
                'date': date,
                'n_positions': allocation.n_positions,
                'exposure': allocation.total_exposure,
                'cash': allocation.cash_weight,
                'portfolio_return': portfolio_return,
                'equity': equity,
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("Testing TDA Portfolio Constructor...")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    
    # Mock StockScores
    stocks = []
    for i in range(50):
        cluster = i // 10
        stocks.append(StockScore(
            ticker=f"STOCK{i:02d}",
            cluster_id=cluster,
            degree_centrality=np.random.rand(),
            eigenvector_centrality=np.random.rand(),
            pagerank=np.random.rand() * 0.1,
            return_5d=np.random.randn() * 0.05,
            return_20d=np.random.randn() * 0.1,
            volatility=0.2 + np.random.rand() * 0.3,
            cluster_score=np.random.rand() * 50,
            momentum_score=np.random.rand() * 50,
            risk_score=np.random.rand() * 50,
            total_score=np.random.rand() * 100,
        ))
    
    # Sort by score
    stocks.sort(key=lambda x: x.total_score, reverse=True)
    
    # Mock returns DataFrame
    returns_df = pd.DataFrame(
        np.random.randn(60, 50) * 0.02,
        columns=[f"STOCK{i:02d}" for i in range(50)]
    )
    
    # Mock cluster analysis
    cluster_analysis = {
        i: ClusterAnalysis(
            cluster_id=i,
            n_stocks=10,
            avg_return_5d=np.random.randn() * 0.03,
            avg_return_20d=np.random.randn() * 0.08,
            avg_volatility=0.25,
            internal_correlation=0.5,
            cluster_momentum=np.random.randn() * 0.5,
            cluster_quality=0.6,
        )
        for i in range(5)
    }
    
    # Test portfolio constructor
    constructor = TDAPortfolioConstructor(
        max_position_weight=0.10,
        max_cluster_weight=0.30,
    )
    
    # Test different methods
    methods = ["equal_weight", "score_weight", "inverse_vol", "cluster_balanced"]
    
    for method in methods:
        print(f"\n{method.upper()} Portfolio:")
        print("-" * 50)
        
        allocation = constructor.construct_portfolio(
            stocks[:30],  # Top 30 stocks
            cluster_analysis,
            returns_df,
            regime="BULL",
            method=method,
        )
        
        print(f"  Positions: {allocation.n_positions}")
        print(f"  Total Exposure: {allocation.total_exposure:.1%}")
        print(f"  Cash: {allocation.cash_weight:.1%}")
        print(f"  Clusters: {allocation.n_clusters}")
        print(f"  Concentration (HHI): {allocation.concentration:.4f}")
        print(f"  Expected Vol: {allocation.expected_vol:.1%}")
        
        if allocation.positions:
            print(f"\n  Top 5 Positions:")
            for p in sorted(allocation.positions, key=lambda x: x.weight, reverse=True)[:5]:
                print(f"    {p.ticker}: {p.weight:.1%} (cluster {p.cluster_id})")
    
    # Test regime-based exposure
    print("\n" + "=" * 50)
    print("REGIME-BASED ALLOCATIONS:")
    print("-" * 50)
    
    for regime in ["BULL", "RECOVERY", "TRANSITION", "BEAR", "CRISIS"]:
        allocation = constructor.construct_portfolio(
            stocks[:30],
            cluster_analysis,
            returns_df,
            regime=regime,
            method="score_weight",
        )
        print(f"  {regime:<12}: {allocation.total_exposure:>6.1%} exposure, {allocation.n_positions} positions")
    
    print("\nTDA Portfolio Constructor tests complete!")
