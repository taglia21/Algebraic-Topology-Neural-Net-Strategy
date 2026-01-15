"""
Regime-Adaptive Strategy for Phase 7.

Detects market regimes and adapts strategy parameters:
- Bull/Bear/Sideways detection using market breadth
- Momentum/TDA weight adjustment by regime
- Position sizing modification
- Dynamic stop-loss levels

Regime indicators:
- Price vs 50/200-day MA
- Market breadth (% stocks above MA)
- Volatility regime (VIX or realized vol)
- Trend strength (ADX-like)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    RECOVERY = "recovery"
    DISTRIBUTION = "distribution"


@dataclass
class RegimeParameters:
    """Strategy parameters for each regime."""
    regime: MarketRegime
    
    # Factor weights
    momentum_weight: float = 0.70
    tda_weight: float = 0.30
    
    # Position sizing
    position_size_mult: float = 1.0
    max_positions: int = 20
    
    # Stop-loss settings
    atr_multiplier: float = 2.0
    trailing_activation: float = 0.05
    
    # Cash allocation
    target_invested_pct: float = 0.95


# Default parameters for each regime
REGIME_PARAMS = {
    MarketRegime.BULL: RegimeParameters(
        regime=MarketRegime.BULL,
        momentum_weight=0.75,
        tda_weight=0.25,
        position_size_mult=1.2,
        max_positions=25,
        atr_multiplier=2.5,
        trailing_activation=0.03,
        target_invested_pct=0.95,
    ),
    MarketRegime.BEAR: RegimeParameters(
        regime=MarketRegime.BEAR,
        momentum_weight=0.40,  # Momentum less reliable in bear
        tda_weight=0.60,  # TDA for structural analysis
        position_size_mult=0.5,
        max_positions=10,
        atr_multiplier=1.5,  # Tighter stops
        trailing_activation=0.02,
        target_invested_pct=0.50,
    ),
    MarketRegime.SIDEWAYS: RegimeParameters(
        regime=MarketRegime.SIDEWAYS,
        momentum_weight=0.50,
        tda_weight=0.50,
        position_size_mult=0.8,
        max_positions=15,
        atr_multiplier=2.0,
        trailing_activation=0.04,
        target_invested_pct=0.75,
    ),
    MarketRegime.HIGH_VOL: RegimeParameters(
        regime=MarketRegime.HIGH_VOL,
        momentum_weight=0.30,
        tda_weight=0.70,  # TDA better for vol analysis
        position_size_mult=0.4,
        max_positions=8,
        atr_multiplier=1.5,
        trailing_activation=0.02,
        target_invested_pct=0.40,
    ),
    MarketRegime.RECOVERY: RegimeParameters(
        regime=MarketRegime.RECOVERY,
        momentum_weight=0.80,  # Strong momentum in recovery
        tda_weight=0.20,
        position_size_mult=1.0,
        max_positions=20,
        atr_multiplier=2.0,
        trailing_activation=0.04,
        target_invested_pct=0.85,
    ),
    MarketRegime.DISTRIBUTION: RegimeParameters(
        regime=MarketRegime.DISTRIBUTION,
        momentum_weight=0.50,
        tda_weight=0.50,
        position_size_mult=0.6,
        max_positions=12,
        atr_multiplier=1.5,
        trailing_activation=0.03,
        target_invested_pct=0.60,
    ),
}


@dataclass
class RegimeIndicators:
    """Raw indicators used for regime detection."""
    price_vs_ma50: float = 0.0  # Current price / 50-day MA
    price_vs_ma200: float = 0.0  # Current price / 200-day MA
    ma50_vs_ma200: float = 0.0  # 50-day MA / 200-day MA
    
    breadth_50: float = 0.0  # % of stocks above 50-day MA
    breadth_200: float = 0.0  # % of stocks above 200-day MA
    
    volatility: float = 0.0  # Realized volatility (annualized)
    volatility_percentile: float = 0.0  # Vol rank vs history
    
    trend_strength: float = 0.0  # ADX-like measure (0-1)
    
    recent_return_20d: float = 0.0  # 20-day return
    recent_return_60d: float = 0.0  # 60-day return


class RegimeDetector:
    """
    Detect market regime from price and breadth data.
    
    Uses multiple indicators:
    - Price relative to moving averages
    - Market breadth
    - Volatility regime
    - Trend strength
    """
    
    def __init__(
        self,
        vol_high_threshold: float = 0.25,  # Annualized vol above this = high vol
        vol_low_threshold: float = 0.12,  # Vol below this = low vol
        breadth_bull_threshold: float = 0.60,  # >60% above MA = bullish
        breadth_bear_threshold: float = 0.40,  # <40% above MA = bearish
        trend_threshold: float = 0.3,  # ADX > 0.3 = trending
    ):
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold
        self.breadth_bull_threshold = breadth_bull_threshold
        self.breadth_bear_threshold = breadth_bear_threshold
        self.trend_threshold = trend_threshold
        
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
    
    def calculate_indicators(
        self,
        market_prices: pd.Series,  # Market index (e.g., SPY)
        stock_prices: Dict[str, pd.Series] = None,  # Individual stocks for breadth
    ) -> RegimeIndicators:
        """
        Calculate regime indicators from price data.
        
        Args:
            market_prices: Series of market index prices
            stock_prices: Dict of stock price series for breadth calculation
            
        Returns:
            RegimeIndicators
        """
        if len(market_prices) < 200:
            return RegimeIndicators()
        
        current_price = market_prices.iloc[-1]
        
        # Moving averages
        ma50 = market_prices.iloc[-50:].mean()
        ma200 = market_prices.iloc[-200:].mean()
        
        price_vs_ma50 = current_price / ma50 if ma50 > 0 else 1.0
        price_vs_ma200 = current_price / ma200 if ma200 > 0 else 1.0
        ma50_vs_ma200 = ma50 / ma200 if ma200 > 0 else 1.0
        
        # Volatility
        returns = market_prices.pct_change().dropna()
        vol_20d = returns.iloc[-20:].std() * np.sqrt(252)
        vol_60d = returns.iloc[-60:].std() * np.sqrt(252)
        volatility = (vol_20d + vol_60d) / 2
        
        # Vol percentile (vs last 252 days)
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = (rolling_vol.iloc[-1] < rolling_vol.iloc[-252:]).mean()
        
        # Recent returns
        ret_20d = (current_price / market_prices.iloc[-20]) - 1 if len(market_prices) >= 20 else 0
        ret_60d = (current_price / market_prices.iloc[-60]) - 1 if len(market_prices) >= 60 else 0
        
        # Trend strength (using price momentum and MA alignment)
        ma_aligned = 1.0 if (price_vs_ma50 > 1 and price_vs_ma200 > 1 and ma50_vs_ma200 > 1) else 0.5
        if price_vs_ma50 < 1 and price_vs_ma200 < 1 and ma50_vs_ma200 < 1:
            ma_aligned = 0.0
        
        # Simple trend strength
        price_range = market_prices.iloc[-50:].max() - market_prices.iloc[-50:].min()
        price_move = abs(current_price - market_prices.iloc[-50])
        trend_strength = (price_move / price_range) if price_range > 0 else 0
        trend_strength = min(1.0, trend_strength * ma_aligned)
        
        # Breadth calculation
        breadth_50 = 0.5
        breadth_200 = 0.5
        
        if stock_prices:
            above_50 = 0
            above_200 = 0
            total = 0
            
            for ticker, prices in stock_prices.items():
                if len(prices) >= 200:
                    stock_ma50 = prices.iloc[-50:].mean()
                    stock_ma200 = prices.iloc[-200:].mean()
                    current = prices.iloc[-1]
                    
                    if current > stock_ma50:
                        above_50 += 1
                    if current > stock_ma200:
                        above_200 += 1
                    total += 1
            
            if total > 0:
                breadth_50 = above_50 / total
                breadth_200 = above_200 / total
        
        return RegimeIndicators(
            price_vs_ma50=price_vs_ma50,
            price_vs_ma200=price_vs_ma200,
            ma50_vs_ma200=ma50_vs_ma200,
            breadth_50=breadth_50,
            breadth_200=breadth_200,
            volatility=volatility,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            recent_return_20d=ret_20d,
            recent_return_60d=ret_60d,
        )
    
    def detect_regime(
        self,
        indicators: RegimeIndicators,
    ) -> MarketRegime:
        """
        Determine current market regime from indicators.
        
        Logic:
        1. High volatility regime takes precedence
        2. Then check trend direction and breadth
        3. Identify recovery/distribution phases
        """
        # High volatility regime
        if indicators.volatility > self.vol_high_threshold:
            return MarketRegime.HIGH_VOL
        
        # Strong bull market
        if (indicators.price_vs_ma200 > 1.05 and 
            indicators.ma50_vs_ma200 > 1.02 and
            indicators.breadth_50 > self.breadth_bull_threshold):
            return MarketRegime.BULL
        
        # Bear market
        if (indicators.price_vs_ma200 < 0.95 and
            indicators.ma50_vs_ma200 < 0.98 and
            indicators.breadth_50 < self.breadth_bear_threshold):
            return MarketRegime.BEAR
        
        # Recovery (coming out of bear)
        if (indicators.price_vs_ma50 > 1.02 and
            indicators.price_vs_ma200 < 1.02 and
            indicators.recent_return_20d > 0.05):
            return MarketRegime.RECOVERY
        
        # Distribution (topping)
        if (indicators.price_vs_ma50 < 1.0 and
            indicators.price_vs_ma200 > 1.0 and
            indicators.recent_return_20d < -0.02):
            return MarketRegime.DISTRIBUTION
        
        # Default: sideways
        return MarketRegime.SIDEWAYS
    
    def get_regime(
        self,
        market_prices: pd.Series,
        stock_prices: Dict[str, pd.Series] = None,
        as_of_date: datetime = None,
    ) -> Tuple[MarketRegime, RegimeIndicators]:
        """
        Get current regime and indicators.
        
        Args:
            market_prices: Market index prices
            stock_prices: Individual stock prices for breadth
            as_of_date: Date for recording history
            
        Returns:
            Tuple of (regime, indicators)
        """
        indicators = self.calculate_indicators(market_prices, stock_prices)
        regime = self.detect_regime(indicators)
        
        # Record in history
        if as_of_date:
            self.regime_history.append((as_of_date, regime))
        
        return regime, indicators
    
    def get_regime_params(self, regime: MarketRegime) -> RegimeParameters:
        """Get parameters for a given regime."""
        return REGIME_PARAMS.get(regime, REGIME_PARAMS[MarketRegime.SIDEWAYS])


class RegimeAdaptiveStrategy:
    """
    Strategy that adapts to market regimes.
    
    Features:
    - Dynamic factor weight adjustment
    - Regime-based position sizing
    - Adaptive stop-loss levels
    """
    
    def __init__(
        self,
        base_momentum_weight: float = 0.70,
        base_tda_weight: float = 0.30,
        base_n_stocks: int = 20,
        regime_adaptation: bool = True,
        smoothing_periods: int = 5,  # Regime smoothing
    ):
        self.base_momentum_weight = base_momentum_weight
        self.base_tda_weight = base_tda_weight
        self.base_n_stocks = base_n_stocks
        self.regime_adaptation = regime_adaptation
        self.smoothing_periods = smoothing_periods
        
        self.detector = RegimeDetector()
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history: List[MarketRegime] = []
    
    def update_regime(
        self,
        market_prices: pd.Series,
        stock_prices: Dict[str, pd.Series] = None,
    ) -> MarketRegime:
        """
        Update current regime with smoothing.
        
        Args:
            market_prices: Market index prices
            stock_prices: Stock prices for breadth
            
        Returns:
            Smoothed current regime
        """
        regime, indicators = self.detector.get_regime(market_prices, stock_prices)
        
        self.regime_history.append(regime)
        if len(self.regime_history) > self.smoothing_periods * 2:
            self.regime_history = self.regime_history[-self.smoothing_periods * 2:]
        
        # Smoothing: require consistent regime detection
        if len(self.regime_history) >= self.smoothing_periods:
            recent = self.regime_history[-self.smoothing_periods:]
            most_common = max(set(recent), key=recent.count)
            if recent.count(most_common) >= self.smoothing_periods // 2 + 1:
                self.current_regime = most_common
        else:
            self.current_regime = regime
        
        return self.current_regime
    
    def get_current_params(self) -> RegimeParameters:
        """Get current regime parameters."""
        if not self.regime_adaptation:
            # Return base parameters as sideways
            return RegimeParameters(
                regime=MarketRegime.SIDEWAYS,
                momentum_weight=self.base_momentum_weight,
                tda_weight=self.base_tda_weight,
                max_positions=self.base_n_stocks,
            )
        
        return self.detector.get_regime_params(self.current_regime)
    
    def get_factor_weights(self) -> Tuple[float, float]:
        """Get current momentum and TDA weights."""
        params = self.get_current_params()
        return params.momentum_weight, params.tda_weight
    
    def get_position_size_multiplier(self) -> float:
        """Get current position size multiplier."""
        params = self.get_current_params()
        return params.position_size_mult
    
    def get_target_positions(self) -> int:
        """Get target number of positions."""
        params = self.get_current_params()
        return params.max_positions
    
    def get_stop_loss_config(self) -> Dict[str, float]:
        """Get current stop-loss configuration."""
        params = self.get_current_params()
        return {
            'atr_multiplier': params.atr_multiplier,
            'trailing_activation': params.trailing_activation,
        }
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state."""
        params = self.get_current_params()
        
        return {
            'current_regime': self.current_regime.value,
            'momentum_weight': params.momentum_weight,
            'tda_weight': params.tda_weight,
            'position_size_mult': params.position_size_mult,
            'max_positions': params.max_positions,
            'target_invested': params.target_invested_pct,
            'atr_multiplier': params.atr_multiplier,
            'regime_history_length': len(self.regime_history),
        }


def print_regime_params() -> None:
    """Print parameters for all regimes."""
    print("\nRegime Parameters:")
    print("="*80)
    print(f"{'Regime':<15} {'Mom Wt':<10} {'TDA Wt':<10} {'Size Mult':<12} {'Max Pos':<10} {'ATR Mult':<10}")
    print("-"*80)
    
    for regime in MarketRegime:
        params = REGIME_PARAMS[regime]
        print(f"{regime.value:<15} {params.momentum_weight:<10.2f} {params.tda_weight:<10.2f} "
              f"{params.position_size_mult:<12.1f} {params.max_positions:<10} {params.atr_multiplier:<10.1f}")


if __name__ == "__main__":
    print("Testing Regime-Adaptive Strategy")
    print("="*50)
    
    # Generate synthetic market data
    np.random.seed(42)
    n = 300
    
    # Create price series with regime changes
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    
    # Bull (0-100), Bear (100-200), Recovery (200-300)
    returns = np.concatenate([
        np.random.randn(100) * 0.01 + 0.001,  # Bull
        np.random.randn(100) * 0.015 - 0.002,  # Bear
        np.random.randn(100) * 0.012 + 0.002,  # Recovery
    ])
    
    prices = pd.Series(
        100 * np.cumprod(1 + returns),
        index=dates,
    )
    
    # Test regime detection
    detector = RegimeDetector()
    
    # Test at different points
    test_points = [99, 199, 299]
    
    for idx in test_points:
        test_prices = prices.iloc[:idx+1]
        regime, indicators = detector.get_regime(test_prices)
        
        print(f"\nDay {idx+1}: Price = ${test_prices.iloc[-1]:.2f}")
        print(f"  Regime: {regime.value}")
        print(f"  Price vs MA50: {indicators.price_vs_ma50:.3f}")
        print(f"  Price vs MA200: {indicators.price_vs_ma200:.3f}")
        print(f"  Volatility: {indicators.volatility:.1%}")
        print(f"  Trend Strength: {indicators.trend_strength:.2f}")
    
    # Print regime parameters
    print_regime_params()
    
    # Test adaptive strategy
    print("\n" + "="*50)
    print("Testing Regime-Adaptive Strategy")
    print("="*50)
    
    strategy = RegimeAdaptiveStrategy(regime_adaptation=True)
    
    # Update through different regimes
    for idx in [99, 199, 299]:
        test_prices = prices.iloc[:idx+1]
        regime = strategy.update_regime(test_prices)
        
        summary = strategy.get_regime_summary()
        print(f"\nDay {idx+1}:")
        print(f"  Regime: {summary['current_regime']}")
        print(f"  Weights: Mom={summary['momentum_weight']:.2f}, TDA={summary['tda_weight']:.2f}")
        print(f"  Size Mult: {summary['position_size_mult']:.1f}")
        print(f"  Max Positions: {summary['max_positions']}")
    
    print("\nRegime-Adaptive Strategy tests complete!")
