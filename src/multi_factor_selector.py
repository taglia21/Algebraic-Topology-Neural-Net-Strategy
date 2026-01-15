"""Multi-Factor Stock Selector for Phase 6.

Combines multiple factors for stock ranking:
- TDA features (topological structure)
- Momentum (12-1 month, 6-month, recent)
- Quality (ROE, debt/equity, earnings stability)
- Value (P/E, P/B, dividend yield)
- Sector diversification constraints
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StockScore:
    """Complete scoring for a single stock."""
    ticker: str
    sector: str
    
    # Component scores (0-1, higher is better)
    momentum_score: float = 0.0
    tda_score: float = 0.0
    quality_score: float = 0.0
    value_score: float = 0.0
    
    # Combined score
    composite_score: float = 0.0
    
    # Ranking
    rank: int = 0
    
    # Metadata
    price: float = 0.0
    avg_volume: float = 0.0
    market_cap: float = 0.0


class MomentumCalculator:
    """Calculate various momentum metrics for stock ranking."""
    
    def __init__(
        self,
        lookback_12m: int = 252,
        lookback_6m: int = 126,
        lookback_1m: int = 21,
        skip_recent: int = 21,  # Skip most recent month (mean reversion)
    ):
        """
        Initialize momentum calculator.
        
        Args:
            lookback_12m: Days for 12-month momentum
            lookback_6m: Days for 6-month momentum
            lookback_1m: Days for 1-month momentum
            skip_recent: Days to skip for 12-1 momentum
        """
        self.lookback_12m = lookback_12m
        self.lookback_6m = lookback_6m
        self.lookback_1m = lookback_1m
        self.skip_recent = skip_recent
    
    def compute_momentum_12_1(self, prices: np.ndarray) -> float:
        """
        Compute 12-month momentum skipping the most recent month.
        
        Classic momentum factor: return over months 2-12.
        """
        if len(prices) < self.lookback_12m:
            return 0.0
        
        # Price 12 months ago
        price_12m = prices[-(self.lookback_12m)]
        # Price 1 month ago (skip recent month)
        price_1m = prices[-self.skip_recent]
        
        if price_12m <= 0:
            return 0.0
        
        return (price_1m / price_12m) - 1.0
    
    def compute_momentum_6m(self, prices: np.ndarray) -> float:
        """Compute 6-month momentum."""
        if len(prices) < self.lookback_6m:
            return 0.0
        
        price_6m = prices[-self.lookback_6m]
        price_now = prices[-1]
        
        if price_6m <= 0:
            return 0.0
        
        return (price_now / price_6m) - 1.0
    
    def compute_momentum_1m(self, prices: np.ndarray) -> float:
        """Compute 1-month momentum (short-term)."""
        if len(prices) < self.lookback_1m:
            return 0.0
        
        price_1m = prices[-self.lookback_1m]
        price_now = prices[-1]
        
        if price_1m <= 0:
            return 0.0
        
        return (price_now / price_1m) - 1.0
    
    def compute_trend_strength(self, prices: np.ndarray) -> float:
        """
        Compute trend strength using price vs moving averages.
        
        Returns value in [-1, 1]:
        - Positive: price above MAs (uptrend)
        - Negative: price below MAs (downtrend)
        """
        if len(prices) < 200:
            return 0.0
        
        price = prices[-1]
        sma_50 = np.mean(prices[-50:])
        sma_200 = np.mean(prices[-200:])
        
        if sma_200 <= 0:
            return 0.0
        
        # Price position relative to MAs
        above_50 = 1.0 if price > sma_50 else -1.0
        above_200 = 1.0 if price > sma_200 else -1.0
        sma_cross = 1.0 if sma_50 > sma_200 else -1.0
        
        # Combine signals
        return (above_50 + above_200 + sma_cross) / 3.0
    
    def compute_composite_momentum(
        self,
        prices: np.ndarray,
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Compute weighted composite momentum score.
        
        Args:
            prices: Price array
            weights: Dict of component weights
            
        Returns:
            Composite momentum score (higher = stronger momentum)
        """
        weights = weights or {
            'momentum_12_1': 0.40,
            'momentum_6m': 0.25,
            'momentum_1m': 0.10,
            'trend_strength': 0.25,
        }
        
        mom_12_1 = self.compute_momentum_12_1(prices)
        mom_6m = self.compute_momentum_6m(prices)
        mom_1m = self.compute_momentum_1m(prices)
        trend = self.compute_trend_strength(prices)
        
        # Weighted sum
        score = (
            weights.get('momentum_12_1', 0.4) * mom_12_1 +
            weights.get('momentum_6m', 0.25) * mom_6m +
            weights.get('momentum_1m', 0.1) * mom_1m +
            weights.get('trend_strength', 0.25) * trend
        )
        
        return score


class MultiFactorSelector:
    """
    Multi-factor stock selector combining TDA, momentum, quality, and value.
    
    Phase 6 implementation for ranking 500+ stocks and selecting top N.
    """
    
    def __init__(
        self,
        factor_weights: Dict[str, float] = None,
        n_stocks: int = 30,
        max_sector_weight: float = 0.30,
        max_single_stock: float = 0.08,
        min_price: float = 5.0,
        min_volume: float = 500_000,
    ):
        """
        Initialize multi-factor selector.
        
        Args:
            factor_weights: Dict of factor weights (should sum to 1.0)
            n_stocks: Number of stocks to select
            max_sector_weight: Maximum weight for any sector
            max_single_stock: Maximum weight for any single stock
            min_price: Minimum stock price
            min_volume: Minimum average volume
        """
        self.factor_weights = factor_weights or {
            'momentum': 0.35,
            'tda': 0.30,
            'quality': 0.20,
            'value': 0.15,
        }
        
        self.n_stocks = n_stocks
        self.max_sector_weight = max_sector_weight
        self.max_single_stock = max_single_stock
        self.min_price = min_price
        self.min_volume = min_volume
        
        self.momentum_calc = MomentumCalculator()
    
    def compute_momentum_scores(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Compute momentum scores for all stocks.
        
        Args:
            ohlcv_dict: Dict mapping ticker to OHLCV DataFrame
            
        Returns:
            Dict mapping ticker to momentum score
        """
        scores = {}
        
        for ticker, df in ohlcv_dict.items():
            try:
                close = df['close'].values if 'close' in df.columns else df['Close'].values
                
                # Need at least 252 days of data
                if len(close) < 252:
                    continue
                
                score = self.momentum_calc.compute_composite_momentum(close)
                scores[ticker] = score
                
            except Exception as e:
                logger.warning(f"Momentum calc failed for {ticker}: {e}")
        
        return scores
    
    def compute_tda_scores(
        self,
        tda_features: Dict[str, Dict[str, float]],
        regime: str = 'neutral',
    ) -> Dict[str, float]:
        """
        Compute TDA-based scores for all stocks.
        
        Args:
            tda_features: Dict mapping ticker to TDA features
            regime: Current market regime
            
        Returns:
            Dict mapping ticker to TDA score
        """
        from src.tda_engine_batched import TDAStockScorer
        
        scorer = TDAStockScorer()
        
        scores = {}
        for ticker, features in tda_features.items():
            scores[ticker] = scorer.score_stock(features, regime)
        
        return scores
    
    def compute_quality_scores(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Compute quality scores based on price stability metrics.
        
        For full production, would use fundamental data (ROE, debt, etc.).
        For Phase 6, we proxy quality with volatility and Sharpe.
        
        Args:
            ohlcv_dict: Dict mapping ticker to OHLCV DataFrame
            
        Returns:
            Dict mapping ticker to quality score
        """
        scores = {}
        
        for ticker, df in ohlcv_dict.items():
            try:
                close = df['close'].values if 'close' in df.columns else df['Close'].values
                
                if len(close) < 252:
                    continue
                
                returns = np.diff(close) / close[:-1]
                
                # Quality proxies
                volatility = np.std(returns) * np.sqrt(252)
                sharpe = (np.mean(returns) * 252) / (volatility + 1e-10)
                
                # Lower volatility and higher Sharpe = higher quality
                # Normalize to [0, 1] range
                vol_score = max(0, 1.0 - volatility / 0.50)  # 50% vol = 0 score
                sharpe_score = min(1.0, max(0, (sharpe + 0.5) / 2.0))  # -0.5 to 1.5 range
                
                scores[ticker] = 0.5 * vol_score + 0.5 * sharpe_score
                
            except Exception as e:
                logger.warning(f"Quality calc failed for {ticker}: {e}")
        
        return scores
    
    def compute_value_scores(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Compute value scores based on price metrics.
        
        For full production, would use P/E, P/B, dividend yield, etc.
        For Phase 6, we use price momentum reversal as a value proxy.
        
        Args:
            ohlcv_dict: Dict mapping ticker to OHLCV DataFrame
            
        Returns:
            Dict mapping ticker to value score
        """
        scores = {}
        
        for ticker, df in ohlcv_dict.items():
            try:
                close = df['close'].values if 'close' in df.columns else df['Close'].values
                
                if len(close) < 252:
                    continue
                
                # Value proxy: stocks that are "cheap" relative to their own history
                # Use percentile rank of current price vs 52-week range
                high_52w = np.max(close[-252:])
                low_52w = np.min(close[-252:])
                current = close[-1]
                
                if high_52w > low_52w:
                    price_pct = (current - low_52w) / (high_52w - low_52w)
                    # Lower percentile = higher value score (contrarian)
                    # But not too low (avoid falling knives)
                    if price_pct < 0.2:  # Below 20th percentile - too risky
                        scores[ticker] = 0.3
                    else:
                        # Sweet spot: 20th to 60th percentile
                        scores[ticker] = max(0, 1.0 - price_pct)
                else:
                    scores[ticker] = 0.5
                    
            except Exception as e:
                logger.warning(f"Value calc failed for {ticker}: {e}")
        
        return scores
    
    def normalize_scores(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range using percentile ranking.
        
        Args:
            scores: Raw scores dict
            
        Returns:
            Normalized scores dict
        """
        if not scores:
            return {}
        
        values = np.array(list(scores.values()))
        tickers = list(scores.keys())
        
        # Percentile rank normalization
        sorted_values = np.argsort(np.argsort(values))
        normalized = sorted_values / (len(values) - 1) if len(values) > 1 else np.zeros_like(values)
        
        return dict(zip(tickers, normalized))
    
    def compute_composite_scores(
        self,
        momentum_scores: Dict[str, float],
        tda_scores: Dict[str, float],
        quality_scores: Dict[str, float],
        value_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute weighted composite scores.
        
        Args:
            momentum_scores: Normalized momentum scores
            tda_scores: Normalized TDA scores
            quality_scores: Normalized quality scores
            value_scores: Normalized value scores
            
        Returns:
            Dict mapping ticker to composite score
        """
        # Get common tickers
        common = set(momentum_scores.keys())
        for scores in [tda_scores, quality_scores, value_scores]:
            common &= set(scores.keys())
        
        composite = {}
        
        for ticker in common:
            score = (
                self.factor_weights['momentum'] * momentum_scores.get(ticker, 0.5) +
                self.factor_weights['tda'] * tda_scores.get(ticker, 0.5) +
                self.factor_weights['quality'] * quality_scores.get(ticker, 0.5) +
                self.factor_weights['value'] * value_scores.get(ticker, 0.5)
            )
            composite[ticker] = score
        
        return composite
    
    def select_stocks(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        tda_features: Dict[str, Dict[str, float]],
        sector_map: Dict[str, str],
        regime: str = 'neutral',
    ) -> List[StockScore]:
        """
        Select top N stocks based on multi-factor scoring with sector diversification.
        
        Args:
            ohlcv_dict: Dict mapping ticker to OHLCV DataFrame
            tda_features: Dict mapping ticker to TDA features
            sector_map: Dict mapping ticker to sector
            regime: Current market regime
            
        Returns:
            List of StockScore objects for selected stocks
        """
        # Compute factor scores
        logger.info("Computing momentum scores...")
        momentum_raw = self.compute_momentum_scores(ohlcv_dict)
        momentum_norm = self.normalize_scores(momentum_raw)
        
        logger.info("Computing TDA scores...")
        tda_raw = self.compute_tda_scores(tda_features, regime)
        tda_norm = self.normalize_scores(tda_raw)
        
        logger.info("Computing quality scores...")
        quality_raw = self.compute_quality_scores(ohlcv_dict)
        quality_norm = self.normalize_scores(quality_raw)
        
        logger.info("Computing value scores...")
        value_raw = self.compute_value_scores(ohlcv_dict)
        value_norm = self.normalize_scores(value_raw)
        
        # Compute composite scores
        composite = self.compute_composite_scores(
            momentum_norm, tda_norm, quality_norm, value_norm
        )
        
        # Rank by composite score
        ranked = sorted(composite.items(), key=lambda x: -x[1])
        
        # Select with sector diversification
        selected = []
        sector_counts = {}
        
        for ticker, score in ranked:
            if len(selected) >= self.n_stocks:
                break
            
            sector = sector_map.get(ticker, 'Unknown')
            current_sector_count = sector_counts.get(sector, 0)
            
            # Check sector constraint
            max_per_sector = int(self.n_stocks * self.max_sector_weight)
            if current_sector_count >= max_per_sector:
                continue
            
            # Get price and volume
            df = ohlcv_dict.get(ticker)
            if df is None or df.empty:
                continue
            
            close = df['close'].values if 'close' in df.columns else df['Close'].values
            volume = df['volume'].values if 'volume' in df.columns else df['Volume'].values
            
            price = close[-1]
            avg_volume = np.mean(volume[-21:])  # 21-day average
            
            # Apply liquidity filters
            if price < self.min_price or avg_volume < self.min_volume:
                continue
            
            # Create score object
            stock_score = StockScore(
                ticker=ticker,
                sector=sector,
                momentum_score=momentum_norm.get(ticker, 0.5),
                tda_score=tda_norm.get(ticker, 0.5),
                quality_score=quality_norm.get(ticker, 0.5),
                value_score=value_norm.get(ticker, 0.5),
                composite_score=score,
                rank=len(selected) + 1,
                price=price,
                avg_volume=avg_volume,
            )
            
            selected.append(stock_score)
            sector_counts[sector] = current_sector_count + 1
        
        logger.info(f"Selected {len(selected)} stocks from {len(composite)} candidates")
        logger.info(f"Sector distribution: {sector_counts}")
        
        return selected
    
    def compute_weights(
        self,
        selected: List[StockScore],
        weighting: str = 'equal',
    ) -> Dict[str, float]:
        """
        Compute portfolio weights for selected stocks.
        
        Args:
            selected: List of StockScore objects
            weighting: 'equal', 'score', or 'inverse_vol'
            
        Returns:
            Dict mapping ticker to weight
        """
        if not selected:
            return {}
        
        weights = {}
        
        if weighting == 'equal':
            w = 1.0 / len(selected)
            for s in selected:
                weights[s.ticker] = w
        
        elif weighting == 'score':
            total = sum(s.composite_score for s in selected)
            for s in selected:
                weights[s.ticker] = s.composite_score / total if total > 0 else 1.0 / len(selected)
        
        else:
            # Default to equal
            w = 1.0 / len(selected)
            for s in selected:
                weights[s.ticker] = w
        
        # Apply max single stock constraint
        for ticker in weights:
            if weights[ticker] > self.max_single_stock:
                weights[ticker] = self.max_single_stock
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MultiFactorSelector...")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    
    def create_ohlcv(n_bars, trend=0.0):
        base_price = 100 + np.cumsum(np.random.randn(n_bars) * 0.5 + trend)
        return pd.DataFrame({
            'open': base_price + np.random.randn(n_bars) * 0.1,
            'high': base_price + np.abs(np.random.randn(n_bars) * 0.5),
            'low': base_price - np.abs(np.random.randn(n_bars) * 0.5),
            'close': base_price,
            'volume': np.random.randint(500000, 2000000, n_bars),
        })
    
    # Create universe
    sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial']
    tickers = [f'STOCK{i}' for i in range(50)]
    
    ohlcv_dict = {}
    sector_map = {}
    
    for i, ticker in enumerate(tickers):
        trend = 0.05 if i < 20 else (-0.02 if i > 40 else 0.01)
        ohlcv_dict[ticker] = create_ohlcv(300, trend)
        sector_map[ticker] = sectors[i % len(sectors)]
    
    # Create fake TDA features
    tda_features = {}
    for ticker in tickers:
        tda_features[ticker] = {
            'persistence_l0': np.random.uniform(0.1, 0.5),
            'persistence_l1': np.random.uniform(0.1, 0.5),
            'entropy_l0': np.random.uniform(0.5, 2.0),
            'entropy_l1': np.random.uniform(0.5, 2.0),
            'max_lifetime_l1': np.random.uniform(0.1, 0.4),
            'sum_lifetime_l0': np.random.uniform(0.5, 2.0),
        }
    
    # Test selector
    selector = MultiFactorSelector(n_stocks=15, max_sector_weight=0.30)
    
    selected = selector.select_stocks(
        ohlcv_dict=ohlcv_dict,
        tda_features=tda_features,
        sector_map=sector_map,
        regime='bull',
    )
    
    print(f"\nSelected {len(selected)} stocks:")
    for s in selected[:10]:
        print(f"  {s.rank:2d}. {s.ticker}: composite={s.composite_score:.3f} "
              f"(mom={s.momentum_score:.2f}, tda={s.tda_score:.2f}, "
              f"qual={s.quality_score:.2f}, val={s.value_score:.2f}) [{s.sector}]")
    
    # Test weights
    weights = selector.compute_weights(selected, weighting='score')
    print(f"\nPortfolio weights (score-based):")
    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1])[:10]:
        print(f"  {ticker}: {weight:.2%}")
    
    print("\n" + "=" * 50)
    print("MultiFactorSelector tests complete!")
