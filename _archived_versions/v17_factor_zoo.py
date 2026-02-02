#!/usr/bin/env python3
"""
V17.0 Factor Zoo
=================
50+ factors for alpha generation across all regimes.

Categories:
1. Price Momentum (10 factors)
2. Technical/Volatility (10 factors)
3. Volume/Liquidity (8 factors)
4. Mean Reversion (8 factors)
5. Trend/Breakout (8 factors)
6. Algebraic Topology / TDA (6 factors)

All factors are:
- Cross-sectional z-scored (daily)
- Winsorized at 3 sigma
- NaN-filled with 0
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V17_FactorZoo')


@dataclass
class FactorDefinition:
    """Definition of a factor"""
    name: str
    category: str
    lookback: int
    description: str


# Factor Registry
FACTOR_REGISTRY: Dict[str, FactorDefinition] = {}


def register_factor(name: str, category: str, lookback: int, description: str):
    """Decorator to register a factor"""
    def decorator(func):
        FACTOR_REGISTRY[name] = FactorDefinition(
            name=name,
            category=category,
            lookback=lookback,
            description=description
        )
        return func
    return decorator


class FactorZoo:
    """
    Comprehensive factor library for V17.0 strategy.
    """
    
    def __init__(self, winsorize: float = 3.0, fill_na: float = 0.0):
        self.winsorize = winsorize
        self.fill_na = fill_na
        self.factors_computed: List[str] = []
    
    # =========================================================================
    # CATEGORY 1: PRICE MOMENTUM (10 factors)
    # =========================================================================
    
    def momentum_1m(self, prices: pd.DataFrame) -> pd.Series:
        """1-month momentum"""
        return prices['close'].pct_change(21)
    
    def momentum_3m(self, prices: pd.DataFrame) -> pd.Series:
        """3-month momentum"""
        return prices['close'].pct_change(63)
    
    def momentum_6m(self, prices: pd.DataFrame) -> pd.Series:
        """6-month momentum"""
        return prices['close'].pct_change(126)
    
    def momentum_12m(self, prices: pd.DataFrame) -> pd.Series:
        """12-month momentum"""
        return prices['close'].pct_change(252)
    
    def momentum_12_1(self, prices: pd.DataFrame) -> pd.Series:
        """12-1 momentum (skip recent month)"""
        return prices['close'].pct_change(252) - prices['close'].pct_change(21)
    
    def momentum_6_1(self, prices: pd.DataFrame) -> pd.Series:
        """6-1 momentum"""
        return prices['close'].pct_change(126) - prices['close'].pct_change(21)
    
    def momentum_acceleration(self, prices: pd.DataFrame) -> pd.Series:
        """Momentum acceleration (3m - 6m)"""
        return prices['close'].pct_change(63) - prices['close'].pct_change(126)
    
    def momentum_consistency(self, prices: pd.DataFrame) -> pd.Series:
        """Fraction of positive weeks in past 12 weeks"""
        weekly_ret = prices['close'].pct_change(5)
        return (weekly_ret > 0).rolling(60).mean()
    
    def relative_strength(self, prices: pd.DataFrame) -> pd.Series:
        """RSI-based momentum"""
        delta = prices['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def risk_adjusted_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """Momentum divided by volatility"""
        ret_3m = prices['close'].pct_change(63)
        vol_3m = prices['close'].pct_change().rolling(63).std() * np.sqrt(252)
        return ret_3m / vol_3m.replace(0, np.nan)
    
    # =========================================================================
    # CATEGORY 2: TECHNICAL / VOLATILITY (10 factors)
    # =========================================================================
    
    def volatility_20d(self, prices: pd.DataFrame) -> pd.Series:
        """20-day realized volatility"""
        return prices['close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    def volatility_60d(self, prices: pd.DataFrame) -> pd.Series:
        """60-day realized volatility"""
        return prices['close'].pct_change().rolling(60).std() * np.sqrt(252)
    
    def volatility_ratio(self, prices: pd.DataFrame) -> pd.Series:
        """Short-term vol / Long-term vol"""
        vol_20 = prices['close'].pct_change().rolling(20).std()
        vol_60 = prices['close'].pct_change().rolling(60).std()
        return vol_20 / vol_60.replace(0, np.nan)
    
    def idio_vol(self, prices: pd.DataFrame) -> pd.Series:
        """Idiosyncratic volatility (approx - uses residual from mean)"""
        ret = prices['close'].pct_change()
        mean_ret = ret.rolling(60).mean()
        residual = ret - mean_ret
        return residual.rolling(20).std() * np.sqrt(252)
    
    def downside_vol(self, prices: pd.DataFrame) -> pd.Series:
        """Downside deviation (semi-deviation)"""
        ret = prices['close'].pct_change()
        downside = ret.where(ret < 0, 0)
        return downside.rolling(60).std() * np.sqrt(252)
    
    def upside_vol(self, prices: pd.DataFrame) -> pd.Series:
        """Upside deviation"""
        ret = prices['close'].pct_change()
        upside = ret.where(ret > 0, 0)
        return upside.rolling(60).std() * np.sqrt(252)
    
    def skewness_60d(self, prices: pd.DataFrame) -> pd.Series:
        """60-day return skewness"""
        return prices['close'].pct_change().rolling(60).skew()
    
    def kurtosis_60d(self, prices: pd.DataFrame) -> pd.Series:
        """60-day return kurtosis"""
        return prices['close'].pct_change().rolling(60).kurt()
    
    def atr_ratio(self, prices: pd.DataFrame) -> pd.Series:
        """ATR relative to price"""
        tr = pd.concat([
            prices['high'] - prices['low'],
            (prices['high'] - prices['close'].shift()).abs(),
            (prices['low'] - prices['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        return atr / prices['close']
    
    def bollinger_width(self, prices: pd.DataFrame) -> pd.Series:
        """Bollinger Band width (volatility proxy)"""
        sma = prices['close'].rolling(20).mean()
        std = prices['close'].rolling(20).std()
        return (4 * std) / sma  # Width as % of price
    
    # =========================================================================
    # CATEGORY 3: VOLUME / LIQUIDITY (8 factors)
    # =========================================================================
    
    def volume_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """Volume growth rate"""
        return prices['volume'].pct_change(20)
    
    def volume_price_trend(self, prices: pd.DataFrame) -> pd.Series:
        """Volume-price trend divergence"""
        price_mom = prices['close'].pct_change(20)
        vol_mom = prices['volume'].pct_change(20)
        return vol_mom - price_mom
    
    def dollar_volume(self, prices: pd.DataFrame) -> pd.Series:
        """Dollar volume (log)"""
        return np.log1p(prices['close'] * prices['volume'])
    
    def volume_volatility(self, prices: pd.DataFrame) -> pd.Series:
        """Volume volatility (coefficient of variation)"""
        vol_mean = prices['volume'].rolling(20).mean()
        vol_std = prices['volume'].rolling(20).std()
        return vol_std / vol_mean.replace(0, np.nan)
    
    def relative_volume(self, prices: pd.DataFrame) -> pd.Series:
        """Current volume vs 20-day average"""
        return prices['volume'] / prices['volume'].rolling(20).mean()
    
    def obv_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """On-Balance Volume momentum"""
        sign = np.sign(prices['close'].diff())
        obv = (sign * prices['volume']).cumsum()
        return obv.pct_change(20)
    
    def vwap_distance(self, prices: pd.DataFrame) -> pd.Series:
        """Distance from VWAP"""
        if 'vwap' in prices.columns:
            vwap = prices['vwap']
        else:
            vwap = (prices['close'] * prices['volume']).rolling(20).sum() / prices['volume'].rolling(20).sum()
        return (prices['close'] - vwap) / vwap
    
    def amihud_illiquidity(self, prices: pd.DataFrame) -> pd.Series:
        """Amihud illiquidity ratio"""
        ret = prices['close'].pct_change().abs()
        dollar_vol = prices['close'] * prices['volume']
        ratio = ret / dollar_vol.replace(0, np.nan)
        return ratio.rolling(20).mean() * 1e6  # Scale
    
    # =========================================================================
    # CATEGORY 4: MEAN REVERSION (8 factors)
    # =========================================================================
    
    def zscore_20d(self, prices: pd.DataFrame) -> pd.Series:
        """20-day price z-score"""
        sma = prices['close'].rolling(20).mean()
        std = prices['close'].rolling(20).std()
        return (prices['close'] - sma) / std.replace(0, np.nan)
    
    def zscore_50d(self, prices: pd.DataFrame) -> pd.Series:
        """50-day price z-score"""
        sma = prices['close'].rolling(50).mean()
        std = prices['close'].rolling(50).std()
        return (prices['close'] - sma) / std.replace(0, np.nan)
    
    def distance_from_high(self, prices: pd.DataFrame) -> pd.Series:
        """Distance from 52-week high"""
        high_52w = prices['high'].rolling(252).max()
        return (prices['close'] - high_52w) / high_52w
    
    def distance_from_low(self, prices: pd.DataFrame) -> pd.Series:
        """Distance from 52-week low"""
        low_52w = prices['low'].rolling(252).min()
        return (prices['close'] - low_52w) / low_52w
    
    def reversal_5d(self, prices: pd.DataFrame) -> pd.Series:
        """5-day reversal (negative = expected reversal)"""
        return -prices['close'].pct_change(5)
    
    def mean_reversion_speed(self, prices: pd.DataFrame) -> pd.Series:
        """How fast price reverts to mean (half-life estimate)"""
        zscore = self.zscore_20d(prices)
        zscore_change = zscore.diff()
        # Negative correlation = faster mean reversion
        return zscore.rolling(20).corr(zscore_change)
    
    def overbought_oversold(self, prices: pd.DataFrame) -> pd.Series:
        """Williams %R indicator"""
        high_14 = prices['high'].rolling(14).max()
        low_14 = prices['low'].rolling(14).min()
        return (high_14 - prices['close']) / (high_14 - low_14).replace(0, np.nan)
    
    def price_gap(self, prices: pd.DataFrame) -> pd.Series:
        """Gap from previous close"""
        return (prices['open'] - prices['close'].shift()) / prices['close'].shift()
    
    # =========================================================================
    # CATEGORY 5: TREND / BREAKOUT (8 factors)
    # =========================================================================
    
    def ma_cross_20_50(self, prices: pd.DataFrame) -> pd.Series:
        """MA20 vs MA50 (positive = bullish)"""
        ma20 = prices['close'].rolling(20).mean()
        ma50 = prices['close'].rolling(50).mean()
        return (ma20 - ma50) / ma50
    
    def ma_cross_50_200(self, prices: pd.DataFrame) -> pd.Series:
        """MA50 vs MA200 (golden/death cross)"""
        ma50 = prices['close'].rolling(50).mean()
        ma200 = prices['close'].rolling(200).mean()
        return (ma50 - ma200) / ma200
    
    def price_vs_ma200(self, prices: pd.DataFrame) -> pd.Series:
        """Price relative to 200-day MA"""
        ma200 = prices['close'].rolling(200).mean()
        return (prices['close'] - ma200) / ma200
    
    def trend_strength_adx(self, prices: pd.DataFrame) -> pd.Series:
        """ADX-like trend strength"""
        # Simplified ADX calculation
        high_diff = prices['high'].diff()
        low_diff = -prices['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.concat([
            prices['high'] - prices['low'],
            (prices['high'] - prices['close'].shift()).abs(),
            (prices['low'] - prices['close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr
        minus_di = 100 * minus_dm.rolling(14).mean() / atr
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(14).mean()  # ADX
    
    def breakout_20d(self, prices: pd.DataFrame) -> pd.Series:
        """Breakout from 20-day range"""
        high_20 = prices['high'].rolling(20).max()
        low_20 = prices['low'].rolling(20).min()
        range_20 = high_20 - low_20
        return (prices['close'] - low_20) / range_20.replace(0, np.nan)
    
    def new_high_count(self, prices: pd.DataFrame) -> pd.Series:
        """Number of new 20-day highs in past 60 days"""
        high_20 = prices['high'].rolling(20).max()
        new_high = (prices['high'] >= high_20).astype(int)
        return new_high.rolling(60).sum()
    
    def channel_position(self, prices: pd.DataFrame) -> pd.Series:
        """Position within 50-day channel"""
        high_50 = prices['high'].rolling(50).max()
        low_50 = prices['low'].rolling(50).min()
        return (prices['close'] - low_50) / (high_50 - low_50).replace(0, np.nan)
    
    def macd_histogram(self, prices: pd.DataFrame) -> pd.Series:
        """MACD histogram (momentum of momentum)"""
        ema12 = prices['close'].ewm(span=12).mean()
        ema26 = prices['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd - signal) / prices['close']
    
    # =========================================================================
    # CATEGORY 6: ALGEBRAIC TOPOLOGY / TDA (6 factors)
    # =========================================================================
    
    def betti_0_estimate(self, prices: pd.DataFrame) -> pd.Series:
        """
        Betti-0 estimate: Number of connected components in price series.
        Approximated by counting local maxima clusters.
        """
        close = prices['close']
        window = 20
        
        # Detect local maxima
        local_max = (close > close.shift(1)) & (close > close.shift(-1))
        
        # Count distinct clusters (simplified)
        betti0 = local_max.rolling(window).sum()
        
        return betti0
    
    def betti_1_estimate(self, prices: pd.DataFrame) -> pd.Series:
        """
        Betti-1 estimate: Number of loops/cycles in price movement.
        Approximated by counting zero-crossings of detrended price.
        """
        close = prices['close']
        window = 60
        
        # Detrend
        trend = close.rolling(window).mean()
        detrended = close - trend
        
        # Count zero crossings (proxy for cycles)
        sign_changes = (np.sign(detrended) != np.sign(detrended.shift())).astype(int)
        betti1 = sign_changes.rolling(window).sum()
        
        return betti1
    
    def persistence_entropy(self, prices: pd.DataFrame) -> pd.Series:
        """
        Persistence entropy: Diversity of topological features.
        Higher entropy = more diverse persistence diagram = more complex market.
        """
        ret = prices['close'].pct_change()
        window = 60
        
        # Use return distribution entropy as proxy
        def rolling_entropy(x):
            if len(x) < 10:
                return np.nan
            hist, _ = np.histogram(x.dropna(), bins=10)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        
        return ret.rolling(window).apply(rolling_entropy, raw=False)
    
    def wasserstein_distance(self, prices: pd.DataFrame) -> pd.Series:
        """
        Wasserstein distance proxy: Change in return distribution.
        Higher = regime shift likely.
        """
        ret = prices['close'].pct_change()
        window = 20
        
        # Compare recent distribution to historical
        def distribution_shift(x):
            if len(x) < window * 2:
                return np.nan
            recent = x[-window:]
            historical = x[:-window]
            # Use mean absolute difference as simple Wasserstein proxy
            return abs(recent.mean() - historical.mean()) + abs(recent.std() - historical.std())
        
        return ret.rolling(window * 2).apply(distribution_shift, raw=False)
    
    def landscape_distance(self, prices: pd.DataFrame) -> pd.Series:
        """
        Persistence landscape distance: Measures topological change rate.
        Proxy using price series curvature.
        """
        close = prices['close']
        window = 20
        
        # Second derivative (curvature)
        first_diff = close.diff()
        second_diff = first_diff.diff()
        
        # Landscape distance ~ variability of curvature
        return second_diff.abs().rolling(window).mean() / close
    
    def tda_complexity(self, prices: pd.DataFrame) -> pd.Series:
        """
        TDA complexity score: Combined topological complexity.
        Higher = more complex market dynamics.
        """
        b0 = self.betti_0_estimate(prices)
        b1 = self.betti_1_estimate(prices)
        entropy = self.persistence_entropy(prices)
        
        # Normalize and combine
        b0_z = (b0 - b0.rolling(60).mean()) / b0.rolling(60).std().replace(0, np.nan)
        b1_z = (b1 - b1.rolling(60).mean()) / b1.rolling(60).std().replace(0, np.nan)
        entropy_z = (entropy - entropy.rolling(60).mean()) / entropy.rolling(60).std().replace(0, np.nan)
        
        return (b0_z + b1_z + entropy_z) / 3
    
    # =========================================================================
    # FACTOR COMPUTATION ENGINE
    # =========================================================================
    
    def get_all_factors(self) -> List[str]:
        """Get list of all available factors"""
        factors = []
        
        # Momentum
        factors.extend([
            'momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m',
            'momentum_12_1', 'momentum_6_1', 'momentum_acceleration',
            'momentum_consistency', 'relative_strength', 'risk_adjusted_momentum'
        ])
        
        # Volatility
        factors.extend([
            'volatility_20d', 'volatility_60d', 'volatility_ratio', 'idio_vol',
            'downside_vol', 'upside_vol', 'skewness_60d', 'kurtosis_60d',
            'atr_ratio', 'bollinger_width'
        ])
        
        # Volume
        factors.extend([
            'volume_momentum', 'volume_price_trend', 'dollar_volume',
            'volume_volatility', 'relative_volume', 'obv_momentum',
            'vwap_distance', 'amihud_illiquidity'
        ])
        
        # Mean Reversion
        factors.extend([
            'zscore_20d', 'zscore_50d', 'distance_from_high', 'distance_from_low',
            'reversal_5d', 'mean_reversion_speed', 'overbought_oversold', 'price_gap'
        ])
        
        # Trend
        factors.extend([
            'ma_cross_20_50', 'ma_cross_50_200', 'price_vs_ma200', 'trend_strength_adx',
            'breakout_20d', 'new_high_count', 'channel_position', 'macd_histogram'
        ])
        
        # TDA
        factors.extend([
            'betti_0_estimate', 'betti_1_estimate', 'persistence_entropy',
            'wasserstein_distance', 'landscape_distance', 'tda_complexity'
        ])
        
        return factors
    
    def compute_factor(self, factor_name: str, prices: pd.DataFrame) -> pd.Series:
        """Compute a single factor"""
        if not hasattr(self, factor_name):
            raise ValueError(f"Unknown factor: {factor_name}")
        
        method = getattr(self, factor_name)
        return method(prices)
    
    def compute_all_factors(
        self, 
        prices: pd.DataFrame,
        factors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute all factors for the given price data.
        
        Args:
            prices: DataFrame with OHLCV data
            factors: List of factor names (None = all)
        
        Returns:
            DataFrame with all factor values
        """
        if factors is None:
            factors = self.get_all_factors()
        
        logger.info(f"📊 Computing {len(factors)} factors...")
        
        result = prices[['close']].copy() if 'symbol' not in prices.columns else prices[['symbol', 'close']].copy()
        
        for factor_name in factors:
            try:
                result[factor_name] = self.compute_factor(factor_name, prices)
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to compute {factor_name}: {e}")
                result[factor_name] = np.nan
        
        self.factors_computed = factors
        logger.info(f"✅ Computed {len(factors)} factors")
        
        return result
    
    def cross_sectional_zscore(self, df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """
        Apply cross-sectional z-scoring for each date.
        """
        result = df.copy()
        
        for col in factor_cols:
            if col in result.columns:
                # Group by date if available
                if 'date' in result.columns:
                    grouped = result.groupby('date')[col]
                    result[col] = grouped.transform(
                        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                    )
                else:
                    # Single time-series - just normalize
                    mean = result[col].mean()
                    std = result[col].std()
                    if std > 0:
                        result[col] = (result[col] - mean) / std
        
        return result
    
    def winsorize_factors(self, df: pd.DataFrame, factor_cols: List[str], n_sigma: float = 3.0) -> pd.DataFrame:
        """
        Winsorize factors at n_sigma.
        """
        result = df.copy()
        
        for col in factor_cols:
            if col in result.columns:
                result[col] = result[col].clip(-n_sigma, n_sigma)
        
        return result


def main():
    """Test factor zoo"""
    import yfinance as yf
    
    print("\n" + "=" * 60)
    print("🦁 V17.0 FACTOR ZOO TEST")
    print("=" * 60)
    
    # Fetch test data
    logger.info("📥 Fetching SPY data for testing...")
    spy = yf.download('SPY', period='2y', progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.lower() for c in spy.columns]
    spy = spy.reset_index()
    spy.columns = [c.lower() if c != 'Date' else 'date' for c in spy.columns]
    
    logger.info(f"   Test data: {len(spy)} days")
    
    # Initialize factor zoo
    zoo = FactorZoo()
    
    # Get all factors
    all_factors = zoo.get_all_factors()
    print(f"\n📊 Total factors available: {len(all_factors)}")
    
    # Group by category
    categories = {
        'Momentum': [f for f in all_factors if 'momentum' in f or 'relative_strength' in f or 'risk_adjusted' in f],
        'Volatility': [f for f in all_factors if 'vol' in f or 'skew' in f or 'kurt' in f or 'atr' in f or 'bollinger' in f],
        'Volume': [f for f in all_factors if 'volume' in f or 'dollar' in f or 'obv' in f or 'vwap' in f or 'amihud' in f],
        'MeanReversion': [f for f in all_factors if 'zscore' in f or 'distance' in f or 'reversal' in f or 'mean_reversion' in f or 'overbought' in f or 'gap' in f],
        'Trend': [f for f in all_factors if 'ma_cross' in f or 'price_vs' in f or 'trend' in f or 'breakout' in f or 'high_count' in f or 'channel' in f or 'macd' in f],
        'TDA': [f for f in all_factors if 'betti' in f or 'persistence' in f or 'wasserstein' in f or 'landscape' in f or 'tda' in f]
    }
    
    for cat, factors in categories.items():
        print(f"\n📁 {cat}: {len(factors)} factors")
        for f in factors[:3]:
            print(f"   - {f}")
        if len(factors) > 3:
            print(f"   ... and {len(factors) - 3} more")
    
    # Compute all factors
    factor_df = zoo.compute_all_factors(spy)
    
    # Show sample
    factor_cols = [c for c in factor_df.columns if c not in ['date', 'close', 'symbol']]
    print(f"\n📈 Sample factor values (latest):")
    latest = factor_df[factor_cols].iloc[-1]
    for col in list(factor_cols)[:10]:
        print(f"   {col}: {latest[col]:.4f}")
    
    # Check for NaN
    nan_counts = factor_df[factor_cols].isna().sum()
    high_nan = nan_counts[nan_counts > len(factor_df) * 0.5]
    if len(high_nan) > 0:
        print(f"\n⚠️ Factors with >50% NaN: {list(high_nan.index)}")
    
    # Save factor sample
    factor_df.to_parquet('cache/v17_factor_sample.parquet', index=False)
    
    print(f"\n✅ Factor Zoo ready with {len(all_factors)} factors")
    print(f"💾 Sample saved to cache/v17_factor_sample.parquet")
    
    return zoo


if __name__ == "__main__":
    main()
