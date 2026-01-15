"""Adaptive Universe Screener for Phase 9.

Implements intelligent stock filtering and dynamic universe construction:
1. Quality screening (fundamental + technical)
2. Momentum-based tiering
3. TDA-enhanced selection
4. Dynamic universe sizing based on market conditions
5. Sector diversification constraints

Target: Select optimal stocks for each regime
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Stock quality classification."""
    ELITE = "elite"       # Top 10% - highest quality
    HIGH = "high"         # Top 25%
    MEDIUM = "medium"     # Middle 50%
    LOW = "low"           # Bottom 25%
    EXCLUDE = "exclude"   # Failed screening


@dataclass
class StockProfile:
    """Complete stock profile for screening."""
    ticker: str
    sector: str
    
    # Fundamental quality
    quality_score: float = 0.0
    quality_tier: QualityTier = QualityTier.MEDIUM
    
    # Momentum metrics
    momentum_score: float = 0.0
    momentum_rank: int = 0
    momentum_percentile: float = 0.5
    
    # TDA metrics  
    tda_score: float = 0.0
    tda_stability: float = 0.0
    
    # Liquidity
    avg_volume: float = 0.0
    avg_dollar_volume: float = 0.0
    price: float = 0.0
    market_cap: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    beta: float = 1.0
    max_drawdown: float = 0.0
    
    # Combined score
    composite_score: float = 0.0
    final_rank: int = 0
    
    # Flags
    passes_screen: bool = True
    excluded_reason: str = ""


@dataclass
class UniverseConfig:
    """Configuration for universe construction."""
    # Size parameters
    target_universe_size: int = 100
    min_universe_size: int = 20
    max_universe_size: int = 200
    
    # Liquidity thresholds
    min_price: float = 5.0
    max_price: float = 10000.0
    min_avg_volume: float = 100_000  # Reduced for faster signal generation
    min_dollar_volume: float = 1_000_000  # Reduced for faster signal generation
    min_trading_days: int = 60  # Reduced from 252 for faster startup
    
    # Quality thresholds
    min_quality_score: float = 0.2  # Relaxed from 0.3
    max_volatility: float = 0.75  # Relaxed from 60% - allow higher vol for higher returns
    max_beta: float = 3.0  # Relaxed from 2.5
    
    # Sector constraints
    max_sector_weight: float = 0.25
    min_sectors: int = 5
    max_single_stock: float = 0.05
    
    # Regime-adaptive sizing
    bull_universe_multiplier: float = 1.2
    bear_universe_multiplier: float = 0.7
    volatile_universe_multiplier: float = 0.5


class LiquidityFilter:
    """Filter stocks by liquidity requirements."""
    
    def __init__(self, config: UniverseConfig):
        self.config = config
    
    def filter(
        self,
        ticker: str,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Check if stock passes liquidity filter.
        
        Returns:
            (passes, reason, metrics)
        """
        if len(prices) < self.config.min_trading_days:
            return False, f"Insufficient history ({len(prices)} days)", {}
        
        current_price = prices[-1]
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        avg_dollar_volume = current_price * avg_volume
        
        metrics = {
            'price': current_price,
            'avg_volume': avg_volume,
            'avg_dollar_volume': avg_dollar_volume,
        }
        
        # Price bounds
        if current_price < self.config.min_price:
            return False, f"Price too low (${current_price:.2f})", metrics
        if current_price > self.config.max_price:
            return False, f"Price too high (${current_price:.2f})", metrics
        
        # Volume requirements
        if avg_volume < self.config.min_avg_volume:
            return False, f"Volume too low ({avg_volume:,.0f})", metrics
        if avg_dollar_volume < self.config.min_dollar_volume:
            return False, f"Dollar volume too low (${avg_dollar_volume:,.0f})", metrics
        
        return True, "", metrics


class QualityScorer:
    """Score stock quality based on fundamental and technical factors."""
    
    def __init__(self, config: UniverseConfig):
        self.config = config
    
    def score(
        self,
        ticker: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        fundamental_data: Optional[Dict] = None,
    ) -> Tuple[float, QualityTier, Dict[str, float]]:
        """
        Compute quality score for a stock.
        
        Returns:
            (quality_score, tier, metrics)
        """
        metrics = {}
        score_components = []
        
        # 1. Volatility quality (lower is better)
        if len(prices) >= 60:
            returns = np.diff(prices[-60:]) / prices[-60:-1]
            volatility = np.std(returns) * np.sqrt(252)
            metrics['volatility'] = volatility
            
            # Score: 1 at 10% vol, 0 at 60% vol
            vol_score = max(0, 1 - (volatility - 0.10) / 0.50)
            score_components.append(('volatility', vol_score, 0.20))
        else:
            vol_score = 0.5
        
        # 2. Drawdown quality (smaller max DD is better)
        if len(prices) >= 252:
            rolling_max = np.maximum.accumulate(prices[-252:])
            drawdowns = prices[-252:] / rolling_max - 1
            max_dd = np.min(drawdowns)
            metrics['max_drawdown'] = max_dd
            
            # Score: 1 at 0% DD, 0 at -50% DD
            dd_score = max(0, 1 + max_dd * 2)
            score_components.append(('drawdown', dd_score, 0.15))
        else:
            dd_score = 0.5
        
        # 3. Trend quality (R² of 60-day regression)
        if len(prices) >= 60:
            from scipy import stats
            x = np.arange(60)
            y = np.log(prices[-60:])
            try:
                _, _, r_value, _, _ = stats.linregress(x, y)
                trend_quality = r_value ** 2
                metrics['trend_quality'] = trend_quality
                score_components.append(('trend', trend_quality, 0.20))
            except:
                trend_quality = 0.5
        else:
            trend_quality = 0.5
        
        # 4. Volume consistency (lower CV is better)
        if len(volumes) >= 60:
            vol_cv = np.std(volumes[-60:]) / (np.mean(volumes[-60:]) + 1e-6)
            metrics['volume_cv'] = vol_cv
            
            # Score: 1 at CV=0.3, 0 at CV=2.0
            vol_consistency = max(0, 1 - (vol_cv - 0.3) / 1.7)
            score_components.append(('vol_consistency', vol_consistency, 0.15))
        else:
            vol_consistency = 0.5
        
        # 5. Price momentum quality (smooth uptrend)
        if len(prices) >= 126:
            returns_6m = prices[-1] / prices[-126] - 1
            metrics['return_6m'] = returns_6m
            
            # Score positive returns higher
            mom_score = np.clip(0.5 + returns_6m, 0, 1)
            score_components.append(('momentum', mom_score, 0.15))
        else:
            mom_score = 0.5
        
        # 6. Beta quality (prefer beta around 1.0)
        # Simplified: use volatility as proxy
        beta_proxy = min(2.5, volatility / 0.20) if 'volatility' in metrics else 1.0
        metrics['beta_proxy'] = beta_proxy
        beta_score = 1 - abs(beta_proxy - 1.0) / 1.5  # Best at beta=1
        score_components.append(('beta', max(0, beta_score), 0.15))
        
        # Compute weighted score
        total_weight = sum(w for _, _, w in score_components)
        quality_score = sum(s * w for _, s, w in score_components) / total_weight
        
        # Determine tier
        if quality_score >= 0.75:
            tier = QualityTier.ELITE
        elif quality_score >= 0.55:
            tier = QualityTier.HIGH
        elif quality_score >= 0.35:
            tier = QualityTier.MEDIUM
        elif quality_score >= self.config.min_quality_score:
            tier = QualityTier.LOW
        else:
            tier = QualityTier.EXCLUDE
        
        return quality_score, tier, metrics


class MomentumRanker:
    """Rank stocks by momentum across multiple horizons."""
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
    ):
        self.weights = weights or {
            'mom_12_1': 0.30,
            'mom_6m': 0.25,
            'mom_3m': 0.20,
            'mom_1m': 0.15,
            'acceleration': 0.10,
        }
    
    def compute_momentum(
        self,
        prices: np.ndarray,
    ) -> Dict[str, float]:
        """Compute multi-horizon momentum."""
        metrics = {}
        
        if len(prices) >= 252:
            metrics['mom_12_1'] = prices[-21] / prices[-252] - 1  # Skip recent month
            metrics['mom_12m'] = prices[-1] / prices[-252] - 1
        else:
            metrics['mom_12_1'] = 0.0
            metrics['mom_12m'] = 0.0
        
        if len(prices) >= 126:
            metrics['mom_6m'] = prices[-1] / prices[-126] - 1
        else:
            metrics['mom_6m'] = 0.0
        
        if len(prices) >= 63:
            metrics['mom_3m'] = prices[-1] / prices[-63] - 1
        else:
            metrics['mom_3m'] = 0.0
        
        if len(prices) >= 21:
            metrics['mom_1m'] = prices[-1] / prices[-21] - 1
        else:
            metrics['mom_1m'] = 0.0
        
        # Acceleration
        if len(prices) >= 42:
            recent = prices[-1] / prices[-21] - 1
            prior = prices[-21] / prices[-42] - 1
            metrics['acceleration'] = recent - prior
        else:
            metrics['acceleration'] = 0.0
        
        return metrics
    
    def score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted momentum score."""
        score = 0.0
        for key, weight in self.weights.items():
            val = metrics.get(key, 0.0)
            # Normalize to roughly -1 to 1
            normalized = np.clip(val * 2, -1, 1)
            score += weight * normalized
        return score
    
    def rank_universe(
        self,
        momentum_scores: Dict[str, float],
    ) -> Dict[str, Dict]:
        """Rank all tickers by momentum."""
        if not momentum_scores:
            return {}
        
        sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_tickers)
        
        results = {}
        for rank, (ticker, score) in enumerate(sorted_tickers, 1):
            results[ticker] = {
                'rank': rank,
                'score': score,
                'percentile': (n - rank + 1) / n,
            }
        
        return results


class TDAScorer:
    """Score stocks based on TDA features."""
    
    def __init__(self):
        self.feature_weights = {
            'persistence_l0': 0.25,
            'persistence_l1': 0.20,
            'turbulence_index': -0.30,  # Lower is better
            'betti_0': -0.15,  # Lower fragmentation is better
            'entropy_h0': -0.10,
        }
    
    def score(
        self,
        tda_features: pd.DataFrame,
        lookback: int = 20,
    ) -> Tuple[float, float]:
        """
        Compute TDA score and stability.
        
        Returns:
            (tda_score, stability)
        """
        if tda_features is None or len(tda_features) < lookback:
            return 0.0, 0.5
        
        recent = tda_features.iloc[-lookback:]
        score_components = []
        
        for feature, weight in self.feature_weights.items():
            if feature in recent.columns:
                val = recent[feature].mean()
                # Normalize roughly
                if 'turbulence' in feature:
                    normalized = (50 - val) / 50  # 0 at turb=50, 1 at turb=0
                elif 'betti' in feature:
                    normalized = 1 - val / 10
                elif 'persistence' in feature:
                    normalized = val * 2  # Scale up
                elif 'entropy' in feature:
                    normalized = 1 - val
                else:
                    normalized = val
                
                # Handle negative weights
                if weight < 0:
                    normalized = -normalized
                    weight = abs(weight)
                
                score_components.append(normalized * weight)
        
        tda_score = sum(score_components) if score_components else 0.0
        
        # Compute stability (inverse of variance)
        if 'turbulence_index' in recent.columns:
            turb_std = recent['turbulence_index'].std()
            stability = 1 / (1 + turb_std / 10)
        else:
            stability = 0.5
        
        return np.clip(tda_score, -1, 1), stability


class SectorDiversifier:
    """Ensure sector diversification in final universe."""
    
    def __init__(self, config: UniverseConfig):
        self.config = config
    
    def diversify(
        self,
        candidates: List[StockProfile],
        target_size: int,
    ) -> List[StockProfile]:
        """
        Select stocks with sector diversification constraints.
        
        Ensures:
        - Max sector_weight% per sector
        - Min min_sectors sectors represented
        - Best stocks within constraints
        """
        if not candidates:
            return []
        
        # Sort by composite score
        sorted_candidates = sorted(candidates, key=lambda x: x.composite_score, reverse=True)
        
        # Track sector counts
        sector_counts: Dict[str, int] = defaultdict(int)
        max_per_sector = int(target_size * self.config.max_sector_weight)
        
        selected = []
        sectors_represented: Set[str] = set()
        
        # First pass: ensure minimum sector diversity
        if self.config.min_sectors > 0:
            sectors_available = set(c.sector for c in sorted_candidates)
            sectors_to_fill = min(self.config.min_sectors, len(sectors_available))
            
            for sector in sectors_available:
                if len(sectors_represented) >= sectors_to_fill:
                    break
                # Get best stock from this sector
                for c in sorted_candidates:
                    if c.sector == sector and c not in selected:
                        selected.append(c)
                        sectors_represented.add(sector)
                        sector_counts[sector] += 1
                        break
        
        # Second pass: fill remaining slots with best stocks
        for candidate in sorted_candidates:
            if len(selected) >= target_size:
                break
            if candidate in selected:
                continue
            
            sector = candidate.sector
            if sector_counts[sector] >= max_per_sector:
                continue
            
            selected.append(candidate)
            sector_counts[sector] += 1
            sectors_represented.add(sector)
        
        return selected


class AdaptiveUniverseScreener:
    """
    Complete adaptive universe screening system.
    
    Dynamically constructs optimal universe based on:
    - Quality/momentum/TDA ranking
    - Market regime
    - Sector diversification
    """
    
    def __init__(
        self,
        config: Optional[UniverseConfig] = None,
    ):
        self.config = config or UniverseConfig()
        
        self.liquidity_filter = LiquidityFilter(self.config)
        self.quality_scorer = QualityScorer(self.config)
        self.momentum_ranker = MomentumRanker()
        self.tda_scorer = TDAScorer()
        self.sector_diversifier = SectorDiversifier(self.config)
        
        # Caches
        self.profiles_cache: Dict[str, StockProfile] = {}
        self.last_universe: List[str] = []
    
    def screen_universe(
        self,
        price_data: Dict[str, np.ndarray],
        volume_data: Dict[str, np.ndarray],
        sector_map: Dict[str, str],
        tda_data: Optional[Dict[str, pd.DataFrame]] = None,
        regime: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, StockProfile]]:
        """
        Screen and rank full universe.
        
        Args:
            price_data: {ticker: price_array}
            volume_data: {ticker: volume_array}
            sector_map: {ticker: sector}
            tda_data: {ticker: tda_features_df}
            regime: Current market regime
            
        Returns:
            (selected_tickers, all_profiles)
        """
        # Determine target size based on regime
        target_size = self._get_target_size(regime)
        
        # Phase 1: Initial screening
        profiles = {}
        for ticker in price_data.keys():
            prices = price_data.get(ticker)
            volumes = volume_data.get(ticker, np.ones_like(prices))
            sector = sector_map.get(ticker, "Other")
            
            if prices is None or len(prices) < 100:
                continue
            
            profile = self._build_profile(
                ticker, prices, volumes, sector,
                tda_data.get(ticker) if tda_data else None,
            )
            
            if profile.passes_screen:
                profiles[ticker] = profile
        
        logger.info(f"Screened {len(profiles)}/{len(price_data)} stocks")
        
        # Phase 2: Compute momentum rankings
        momentum_scores = {}
        for ticker, profile in profiles.items():
            momentum_scores[ticker] = profile.momentum_score
        
        momentum_ranks = self.momentum_ranker.rank_universe(momentum_scores)
        
        for ticker, ranks in momentum_ranks.items():
            if ticker in profiles:
                profiles[ticker].momentum_rank = ranks['rank']
                profiles[ticker].momentum_percentile = ranks['percentile']
        
        # Phase 3: Compute composite scores
        for ticker, profile in profiles.items():
            profile.composite_score = self._compute_composite_score(profile, regime)
        
        # Phase 4: Apply sector diversification
        candidates = list(profiles.values())
        selected = self.sector_diversifier.diversify(candidates, target_size)
        
        # Assign final ranks
        for i, profile in enumerate(selected):
            profile.final_rank = i + 1
        
        # Store results
        selected_tickers = [p.ticker for p in selected]
        self.profiles_cache = profiles
        self.last_universe = selected_tickers
        
        logger.info(f"Selected {len(selected_tickers)} stocks from {len(profiles)} candidates")
        
        return selected_tickers, profiles
    
    def _build_profile(
        self,
        ticker: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        sector: str,
        tda_features: Optional[pd.DataFrame],
    ) -> StockProfile:
        """Build complete stock profile."""
        profile = StockProfile(ticker=ticker, sector=sector)
        
        # Liquidity filter
        passes, reason, liq_metrics = self.liquidity_filter.filter(ticker, prices, volumes)
        if not passes:
            profile.passes_screen = False
            profile.excluded_reason = reason
            profile.quality_tier = QualityTier.EXCLUDE
            return profile
        
        profile.price = liq_metrics.get('price', 0)
        profile.avg_volume = liq_metrics.get('avg_volume', 0)
        profile.avg_dollar_volume = liq_metrics.get('avg_dollar_volume', 0)
        
        # Quality scoring
        quality_score, tier, quality_metrics = self.quality_scorer.score(ticker, prices, volumes)
        profile.quality_score = quality_score
        profile.quality_tier = tier
        profile.volatility = quality_metrics.get('volatility', 0.20)
        profile.max_drawdown = quality_metrics.get('max_drawdown', 0)
        
        if tier == QualityTier.EXCLUDE:
            profile.passes_screen = False
            profile.excluded_reason = "Quality too low"
            return profile
        
        # Volatility check
        if profile.volatility > self.config.max_volatility:
            profile.passes_screen = False
            profile.excluded_reason = f"Volatility too high ({profile.volatility:.1%})"
            profile.quality_tier = QualityTier.EXCLUDE
            return profile
        
        # Momentum scoring
        mom_metrics = self.momentum_ranker.compute_momentum(prices)
        profile.momentum_score = self.momentum_ranker.score(mom_metrics)
        
        # TDA scoring
        tda_score, tda_stability = self.tda_scorer.score(tda_features)
        profile.tda_score = tda_score
        profile.tda_stability = tda_stability
        
        return profile
    
    def _compute_composite_score(
        self,
        profile: StockProfile,
        regime: Optional[str] = None,
    ) -> float:
        """Compute final composite score."""
        # Base weights
        weights = {
            'quality': 0.20,
            'momentum': 0.40,
            'tda': 0.25,
            'tier_bonus': 0.15,
        }
        
        # Adjust weights by regime
        if regime in ['bull_momentum', 'low_volatility']:
            weights['momentum'] = 0.50
            weights['quality'] = 0.15
        elif regime in ['bear_defensive', 'high_volatility']:
            weights['quality'] = 0.35
            weights['momentum'] = 0.25
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        # Compute score
        score = 0.0
        score += weights['quality'] * profile.quality_score
        score += weights['momentum'] * (0.5 + profile.momentum_score * 0.5)  # Scale to 0-1
        score += weights['tda'] * (0.5 + profile.tda_score * 0.5)
        
        # Tier bonus
        tier_bonuses = {
            QualityTier.ELITE: 1.0,
            QualityTier.HIGH: 0.7,
            QualityTier.MEDIUM: 0.4,
            QualityTier.LOW: 0.1,
        }
        score += weights['tier_bonus'] * tier_bonuses.get(profile.quality_tier, 0)
        
        return score
    
    def _get_target_size(self, regime: Optional[str]) -> int:
        """Get target universe size based on regime."""
        base = self.config.target_universe_size
        
        if regime is None:
            return base
        
        if regime in ['bull_momentum', 'low_volatility']:
            multiplier = self.config.bull_universe_multiplier
        elif regime in ['bear_defensive']:
            multiplier = self.config.bear_universe_multiplier
        elif regime in ['high_volatility']:
            multiplier = self.config.volatile_universe_multiplier
        else:
            multiplier = 1.0
        
        target = int(base * multiplier)
        return max(self.config.min_universe_size, min(self.config.max_universe_size, target))
    
    def get_universe_summary(self) -> Dict:
        """Get summary of current universe."""
        if not self.last_universe:
            return {'count': 0}
        
        profiles = [self.profiles_cache[t] for t in self.last_universe if t in self.profiles_cache]
        
        # Sector breakdown
        sector_counts = defaultdict(int)
        for p in profiles:
            sector_counts[p.sector] += 1
        
        # Quality breakdown
        tier_counts = defaultdict(int)
        for p in profiles:
            tier_counts[p.quality_tier.value] += 1
        
        return {
            'count': len(profiles),
            'sectors': dict(sector_counts),
            'quality_tiers': dict(tier_counts),
            'avg_quality': np.mean([p.quality_score for p in profiles]),
            'avg_momentum': np.mean([p.momentum_score for p in profiles]),
            'avg_tda': np.mean([p.tda_score for p in profiles]),
        }
