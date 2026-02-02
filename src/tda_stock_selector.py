"""TDA-Based Stock Selection.

Phase 5: Selects stocks based on topological cluster analysis.
Uses graph centrality, cluster momentum, and TDA features to rank stocks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


@dataclass
class StockScore:
    """Score for a single stock."""
    ticker: str
    cluster_id: int
    
    # Centrality metrics
    degree_centrality: float
    eigenvector_centrality: float
    pagerank: float
    
    # Momentum metrics
    return_5d: float
    return_20d: float
    volatility: float
    
    # Combined scores
    cluster_score: float    # How well connected within cluster
    momentum_score: float   # Momentum ranking
    risk_score: float       # Risk-adjusted score
    total_score: float      # Final composite score


@dataclass
class ClusterAnalysis:
    """Analysis of a single cluster."""
    cluster_id: int
    n_stocks: int
    avg_return_5d: float
    avg_return_20d: float
    avg_volatility: float
    internal_correlation: float
    cluster_momentum: float
    cluster_quality: float


class TDAStockSelector:
    """
    Stock selection using topological data analysis.
    
    Strategy:
    1. Build correlation graph from stock returns
    2. Detect clusters using community detection
    3. Rank clusters by momentum and quality
    4. Within top clusters, rank stocks by centrality
    5. Select top N stocks with best combined scores
    
    Key Insights:
    - Central stocks in good clusters tend to lead moves
    - High eigenvector centrality = connected to important stocks
    - PageRank identifies stocks that matter in the network
    """
    
    def __init__(
        self,
        n_stocks: int = 30,
        min_correlation: float = 0.3,
        lookback_momentum: int = 20,
        top_clusters_pct: float = 0.8,  # Select from top 80% of clusters (broader selection)
    ):
        """
        Initialize stock selector.
        
        Args:
            n_stocks: Number of stocks to select
            min_correlation: Minimum correlation for graph edges
            lookback_momentum: Days for momentum calculation
            top_clusters_pct: Percentage of top clusters to consider
        """
        self.n_stocks = n_stocks
        self.min_correlation = min_correlation
        self.lookback_momentum = lookback_momentum
        self.top_clusters_pct = top_clusters_pct
    
    def build_correlation_graph(
        self,
        returns_df: pd.DataFrame,
        lookback: int = 30,
    ) -> 'nx.Graph':
        """
        Build correlation graph from returns data.
        
        Args:
            returns_df: Daily returns DataFrame
            lookback: Days for correlation calculation
            
        Returns:
            NetworkX graph with stocks as nodes and correlations as edge weights
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for graph operations")
        
        # Calculate correlation
        corr = returns_df.iloc[-lookback:].corr()
        
        # Build graph
        G = nx.Graph()
        tickers = returns_df.columns.tolist()
        
        for ticker in tickers:
            G.add_node(ticker)
        
        for i, ticker_i in enumerate(tickers):
            for j, ticker_j in enumerate(tickers):
                if i < j:
                    weight = corr.loc[ticker_i, ticker_j]
                    if not np.isnan(weight) and weight > self.min_correlation:
                        G.add_edge(ticker_i, ticker_j, weight=weight)
        
        return G
    
    def detect_clusters(
        self,
        G: 'nx.Graph',
    ) -> Dict[str, int]:
        """
        Detect clusters using Louvain community detection.
        
        Args:
            G: Correlation graph
            
        Returns:
            Dict mapping ticker to cluster ID
        """
        from networkx.algorithms.community import louvain_communities
        
        communities = louvain_communities(G, weight='weight', seed=42)
        
        ticker_to_cluster = {}
        for cluster_id, community in enumerate(communities):
            for ticker in community:
                ticker_to_cluster[ticker] = cluster_id
        
        # Handle isolated nodes
        for node in G.nodes():
            if node not in ticker_to_cluster:
                ticker_to_cluster[node] = -1  # Unclustered
        
        return ticker_to_cluster
    
    def compute_centrality_metrics(
        self,
        G: 'nx.Graph',
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Compute centrality metrics for all nodes.
        
        Args:
            G: Correlation graph
            
        Returns:
            Tuple of (degree_centrality, eigenvector_centrality, pagerank)
        """
        # Degree centrality
        degree = nx.degree_centrality(G)
        
        # Eigenvector centrality (may fail on disconnected graphs)
        try:
            eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=500)
        except:
            eigenvector = {node: 1.0 / len(G.nodes()) for node in G.nodes()}
        
        # PageRank
        try:
            pagerank = nx.pagerank(G, weight='weight')
        except:
            pagerank = {node: 1.0 / len(G.nodes()) for node in G.nodes()}
        
        return degree, eigenvector, pagerank
    
    def compute_momentum_metrics(
        self,
        returns_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute momentum metrics for all stocks.
        
        Args:
            returns_df: Daily returns DataFrame
            
        Returns:
            Dict mapping ticker to momentum metrics
        """
        metrics = {}
        
        for ticker in returns_df.columns:
            returns = returns_df[ticker].dropna()
            
            if len(returns) < 5:
                continue
            
            # 5-day return
            ret_5d = (1 + returns.iloc[-5:]).prod() - 1 if len(returns) >= 5 else 0
            
            # 20-day return
            ret_20d = (1 + returns.iloc[-20:]).prod() - 1 if len(returns) >= 20 else ret_5d
            
            # 60-day return (medium-term momentum)
            ret_60d = (1 + returns.iloc[-60:]).prod() - 1 if len(returns) >= 60 else ret_20d
            
            # Volatility (20-day)
            vol = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else returns.std() * np.sqrt(252)
            
            metrics[ticker] = {
                'return_5d': ret_5d,
                'return_20d': ret_20d,
                'return_60d': ret_60d,
                'volatility': vol if not np.isnan(vol) else 0.3,
            }
        
        return metrics
    
    def analyze_clusters(
        self,
        returns_df: pd.DataFrame,
        ticker_to_cluster: Dict[str, int],
        corr_matrix: pd.DataFrame,
    ) -> Dict[int, ClusterAnalysis]:
        """
        Analyze each cluster's characteristics.
        
        Args:
            returns_df: Daily returns DataFrame
            ticker_to_cluster: Ticker to cluster mapping
            corr_matrix: Correlation matrix
            
        Returns:
            Dict mapping cluster_id to ClusterAnalysis
        """
        clusters = {}
        
        # Group tickers by cluster
        cluster_tickers = {}
        for ticker, cluster_id in ticker_to_cluster.items():
            if cluster_id not in cluster_tickers:
                cluster_tickers[cluster_id] = []
            cluster_tickers[cluster_id].append(ticker)
        
        for cluster_id, tickers in cluster_tickers.items():
            if cluster_id == -1:  # Skip unclustered
                continue
            
            if len(tickers) < 3:
                continue
            
            # Calculate cluster metrics
            valid_tickers = [t for t in tickers if t in returns_df.columns]
            if len(valid_tickers) < 3:
                continue
            
            cluster_returns = returns_df[valid_tickers]
            
            # Average returns
            avg_5d = cluster_returns.iloc[-5:].mean().mean() * 5
            avg_20d = cluster_returns.iloc[-20:].mean().mean() * 20
            
            # Average volatility
            avg_vol = cluster_returns.iloc[-20:].std().mean() * np.sqrt(252)
            
            # Internal correlation
            cluster_corr = corr_matrix.loc[valid_tickers, valid_tickers]
            internal_corr = cluster_corr.values[np.triu_indices(len(valid_tickers), k=1)].mean()
            
            # Cluster momentum (relative strength)
            cluster_momentum = avg_20d / max(avg_vol, 0.1)
            
            # Cluster quality (high internal correlation + momentum)
            cluster_quality = internal_corr * 0.5 + min(max(cluster_momentum, -1), 1) * 0.5
            
            clusters[cluster_id] = ClusterAnalysis(
                cluster_id=cluster_id,
                n_stocks=len(valid_tickers),
                avg_return_5d=avg_5d,
                avg_return_20d=avg_20d,
                avg_volatility=avg_vol,
                internal_correlation=internal_corr,
                cluster_momentum=cluster_momentum,
                cluster_quality=cluster_quality,
            )
        
        return clusters
    
    def score_stocks(
        self,
        returns_df: pd.DataFrame,
        G: 'nx.Graph',
        ticker_to_cluster: Dict[str, int],
        cluster_analysis: Dict[int, ClusterAnalysis],
        regime: str = "BULL",
    ) -> List[StockScore]:
        """
        Score all stocks based on TDA and momentum metrics.
        
        Args:
            returns_df: Daily returns DataFrame
            G: Correlation graph
            ticker_to_cluster: Cluster assignments
            cluster_analysis: Cluster analysis results
            regime: Current market regime
            
        Returns:
            List of StockScore sorted by total_score descending
        """
        # Get centrality metrics
        degree, eigenvector, pagerank = self.compute_centrality_metrics(G)
        
        # Get momentum metrics
        momentum = self.compute_momentum_metrics(returns_df)
        
        # Get top clusters (by momentum in BULL, by quality in BEAR)
        if regime in ["BULL", "RECOVERY"]:
            sorted_clusters = sorted(
                cluster_analysis.items(),
                key=lambda x: x[1].cluster_momentum,
                reverse=True
            )
        else:
            sorted_clusters = sorted(
                cluster_analysis.items(),
                key=lambda x: -x[1].avg_volatility  # Low vol in BEAR
            )
        
        n_top_clusters = max(1, int(len(sorted_clusters) * self.top_clusters_pct))
        top_cluster_ids = set(c[0] for c in sorted_clusters[:n_top_clusters])
        
        # Score each stock
        scores = []
        
        for ticker in returns_df.columns:
            if ticker not in ticker_to_cluster:
                continue
            
            cluster_id = ticker_to_cluster[ticker]
            
            # In BULL/RECOVERY, don't filter by cluster - let momentum decide
            # Only filter by cluster in BEAR/CRISIS for defensive positioning
            if regime not in ["BULL", "RECOVERY"]:
                if cluster_id not in top_cluster_ids:
                    continue
            
            if ticker not in momentum:
                continue
            
            # Get metrics
            deg_cent = degree.get(ticker, 0)
            eig_cent = eigenvector.get(ticker, 0)
            pr = pagerank.get(ticker, 0)
            mom = momentum[ticker]
            cluster = cluster_analysis.get(cluster_id)
            
            # Cluster score (based on centrality within graph)
            cluster_score = (deg_cent * 0.3 + eig_cent * 0.4 + pr * 0.3) * 100
            
            # Momentum score - multi-timeframe momentum
            # Balanced multi-timeframe approach
            momentum_score = (
                mom['return_60d'] * 50 +  # 60d return (medium-term trend)
                mom['return_20d'] * 35 +  # 20d return (short-term momentum)
                mom['return_5d'] * 15     # 5d return (recent momentum)
            )
            
            # Risk score (penalize high volatility)
            risk_score = max(0, 100 - mom['volatility'] * 100)
            
            # Total score (adjust weights by regime)
            # BULL/RECOVERY: Pure momentum strategy - 85% weight
            if regime in ["BULL", "RECOVERY"]:
                total_score = cluster_score * 0.0 + momentum_score * 0.85 + risk_score * 0.15
            elif regime == "BEAR":
                # In BEAR, prioritize low volatility and defensive positions
                total_score = cluster_score * 0.1 + momentum_score * 0.3 + risk_score * 0.6
            else:  # TRANSITION
cat src/options_engine.py
                total_score = cluster_score * 0.2 + momentum_score * 0.5 + risk_score * 0.3
            
            # Bonus for being in strong cluster
            if cluster and cluster.cluster_quality > 0.5:
                total_score *= 1.2
            
            scores.append(StockScore(
                ticker=ticker,
                cluster_id=cluster_id,
                degree_centrality=deg_cent,
                eigenvector_centrality=eig_cent,
                pagerank=pr,
                return_5d=mom['return_5d'],
                return_20d=mom['return_20d'],
                volatility=mom['volatility'],
                cluster_score=cluster_score,
                momentum_score=momentum_score,
                risk_score=risk_score,
                total_score=total_score,
            ))
        
        # Sort by total score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return scores
    
    def select_stocks(
        self,
        returns_df: pd.DataFrame,
        regime: str = "BULL",
    ) -> Tuple[List[StockScore], Dict[int, ClusterAnalysis]]:
        """
        Main method to select top N stocks.
        
        Args:
            returns_df: Daily returns DataFrame (rows=dates, cols=tickers)
            regime: Current market regime
            
        Returns:
            Tuple of (selected stocks, cluster analysis)
        """
        logger.info(f"Selecting {self.n_stocks} stocks from {len(returns_df.columns)} candidates")
        
        # Build correlation graph
        G = self.build_correlation_graph(returns_df)
        logger.info(f"Built graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Detect clusters
        ticker_to_cluster = self.detect_clusters(G)
        n_clusters = len(set(ticker_to_cluster.values()) - {-1})
        logger.info(f"Detected {n_clusters} clusters")
        
        # Analyze clusters
        corr_matrix = returns_df.iloc[-30:].corr()
        cluster_analysis = self.analyze_clusters(returns_df, ticker_to_cluster, corr_matrix)
        
        # Score stocks
        all_scores = self.score_stocks(
            returns_df, G, ticker_to_cluster, cluster_analysis, regime
        )
        
        # Select top N
        selected = all_scores[:self.n_stocks]
        
        logger.info(f"Selected {len(selected)} stocks")
        for i, s in enumerate(selected[:5]):
            logger.info(f"  {i+1}. {s.ticker}: score={s.total_score:.1f}, cluster={s.cluster_id}")
        
        return selected, cluster_analysis
    
    def get_sector_diversification(
        self,
        selected_stocks: List[StockScore],
        sector_map: Dict[str, str] = None,
    ) -> Dict[str, List[str]]:
        """
        Check sector diversification of selected stocks.
        
        Args:
            selected_stocks: Selected stock scores
            sector_map: Optional ticker to sector mapping
            
        Returns:
            Dict mapping sector to tickers
        """
        if sector_map is None:
            # Group by cluster as proxy for sector
            sector_map = {s.ticker: f"Cluster_{s.cluster_id}" for s in selected_stocks}
        
        sectors = {}
        for s in selected_stocks:
            sector = sector_map.get(s.ticker, "Unknown")
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(s.ticker)
        
        return sectors


if __name__ == "__main__":
    print("Testing TDA Stock Selector...")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    n_stocks = 200
    n_days = 60
    
    # Generate returns with cluster structure
    n_clusters = 5
    cluster_size = n_stocks // n_clusters
    
    returns_list = []
    tickers = []
    
    for c in range(n_clusters):
        # Cluster factor
        cluster_factor = np.random.randn(n_days) * 0.015
        
        # Add momentum to some clusters
        if c < 2:  # First 2 clusters have positive momentum
            cluster_factor += 0.001
        elif c > 3:  # Last cluster has negative momentum
            cluster_factor -= 0.001
        
        for i in range(cluster_size):
            stock_returns = cluster_factor + np.random.randn(n_days) * 0.02
            returns_list.append(stock_returns)
            tickers.append(f"STOCK{c}_{i:02d}")
    
    returns_df = pd.DataFrame(np.array(returns_list).T, columns=tickers)
    
    print(f"Test data: {n_stocks} stocks, {n_days} days, {n_clusters} true clusters")
    
    # Test stock selector
    selector = TDAStockSelector(n_stocks=30, min_correlation=0.3)
    
    selected, cluster_analysis = selector.select_stocks(returns_df, regime="BULL")
    
    print(f"\nTop 10 Selected Stocks (BULL regime):")
    print("-" * 80)
    print(f"{'Rank':<5} {'Ticker':<12} {'Cluster':<8} {'Score':<8} {'Return20d':<10} {'Vol':<8}")
    print("-" * 80)
    
    for i, s in enumerate(selected[:10]):
        print(f"{i+1:<5} {s.ticker:<12} {s.cluster_id:<8} {s.total_score:<8.1f} {s.return_20d*100:<10.1f}% {s.volatility:<8.2f}")
    
    print(f"\nCluster Analysis:")
    print("-" * 80)
    for cid, analysis in sorted(cluster_analysis.items()):
        print(f"Cluster {cid}: {analysis.n_stocks} stocks, "
              f"ret20d={analysis.avg_return_20d*100:.1f}%, "
              f"momentum={analysis.cluster_momentum:.2f}, "
              f"quality={analysis.cluster_quality:.2f}")
    
    print("\nSector/Cluster Diversification:")
    diversification = selector.get_sector_diversification(selected)
    for sector, stocks in diversification.items():
        print(f"  {sector}: {len(stocks)} stocks")
    
    print("\nTDA Stock Selector tests complete!")
