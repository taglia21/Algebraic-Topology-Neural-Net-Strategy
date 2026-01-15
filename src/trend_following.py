"""Trend Following Overlay with Momentum Features.

Phase 4 Optimization: Add trend-following mode to capture directional moves.
When trend is strong (ADX > 25, 50MA > 200MA), follow the trend.
When no trend, use mean reversion signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class TrendMode(Enum):
    """Market trend mode."""
    STRONG_BULL = "strong_bull"   # Strong uptrend - max long
    BULL = "bull"                 # Uptrend - lean long
    NEUTRAL = "neutral"           # No clear trend - mean reversion
    BEAR = "bear"                 # Downtrend - lean short/cash
    STRONG_BEAR = "strong_bear"   # Strong downtrend - max defensive


@dataclass
class TrendSignal:
    """Result of trend analysis."""
    mode: TrendMode
    strength: float  # 0-1 trend strength
    direction: int   # 1=bullish, -1=bearish, 0=neutral
    adx: float       # ADX value
    roc_20: float    # 20-day rate of change
    ma_alignment: int  # 1=bullish, -1=bearish, 0=neutral
    volume_trend: float  # Volume acceleration
    reason: str


class TrendFollowingOverlay:
    """
    Trend-following overlay to enhance TDA signals.
    
    Uses multiple confirmations:
    1. Moving average alignment (50 > 200 for bullish)
    2. ADX for trend strength
    3. Rate of Change for momentum
    4. Volume confirmation
    
    Phase 4: When trend is strong, FOLLOW IT. Don't fight the trend.
    """
    
    def __init__(
        self,
        short_ma: int = 50,
        long_ma: int = 200,
        adx_period: int = 14,
        adx_threshold_strong: float = 30,
        adx_threshold_weak: float = 20,
        roc_period: int = 20,
        volume_period: int = 20
    ):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.adx_period = adx_period
        self.adx_threshold_strong = adx_threshold_strong
        self.adx_threshold_weak = adx_threshold_weak
        self.roc_period = roc_period
        self.volume_period = volume_period
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed averages (Wilder's smoothing)
        period = self.adx_period
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def calculate_roc(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change."""
        return (series / series.shift(period) - 1) * 100
    
    def calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram."""
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_volume_acceleration(self, volume: pd.Series) -> pd.Series:
        """Calculate volume acceleration (current vs average)."""
        avg_volume = volume.rolling(self.volume_period).mean()
        return volume / avg_volume
    
    def analyze_trend(self, df: pd.DataFrame) -> TrendSignal:
        """
        Analyze current trend mode.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            TrendSignal with current trend analysis
        """
        if len(df) < self.long_ma + 10:
            return TrendSignal(
                mode=TrendMode.NEUTRAL,
                strength=0,
                direction=0,
                adx=0,
                roc_20=0,
                ma_alignment=0,
                volume_trend=1.0,
                reason="Insufficient data for trend analysis"
            )
        
        close = df['Close']
        volume = df['Volume']
        
        # Moving averages
        ma_short = close.rolling(self.short_ma).mean()
        ma_long = close.rolling(self.long_ma).mean()
        
        current_price = close.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        # MA alignment
        if current_ma_short > current_ma_long * 1.01:  # 1% buffer
            ma_alignment = 1  # Bullish
        elif current_ma_short < current_ma_long * 0.99:
            ma_alignment = -1  # Bearish
        else:
            ma_alignment = 0  # Neutral
        
        # Price position relative to MAs
        price_above_short = current_price > current_ma_short
        price_above_long = current_price > current_ma_long
        
        # ADX
        adx = self.calculate_adx(df)
        current_adx = adx.iloc[-1]
        
        # Rate of change
        roc_20 = self.calculate_roc(close, self.roc_period).iloc[-1]
        roc_10 = self.calculate_roc(close, 10).iloc[-1]
        roc_50 = self.calculate_roc(close, 50).iloc[-1]
        
        # MACD
        macd, signal, histogram = self.calculate_macd(close)
        macd_bullish = histogram.iloc[-1] > 0
        
        # Volume
        vol_accel = self.calculate_volume_acceleration(volume).iloc[-1]
        
        # Determine trend mode
        bullish_signals = 0
        bearish_signals = 0
        
        if ma_alignment > 0:
            bullish_signals += 1
        elif ma_alignment < 0:
            bearish_signals += 1
        
        if roc_20 > 2:  # Strong positive momentum
            bullish_signals += 1
        elif roc_20 < -2:
            bearish_signals += 1
        
        if price_above_short and price_above_long:
            bullish_signals += 1
        elif not price_above_short and not price_above_long:
            bearish_signals += 1
        
        if macd_bullish:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Trend strength based on ADX
        if current_adx >= self.adx_threshold_strong:
            trend_strength = 1.0
        elif current_adx >= self.adx_threshold_weak:
            trend_strength = 0.5
        else:
            trend_strength = 0.2
        
        # Determine mode
        net_signal = bullish_signals - bearish_signals
        
        if current_adx >= self.adx_threshold_strong:
            if net_signal >= 3:
                mode = TrendMode.STRONG_BULL
                direction = 1
            elif net_signal <= -3:
                mode = TrendMode.STRONG_BEAR
                direction = -1
            elif net_signal > 0:
                mode = TrendMode.BULL
                direction = 1
            elif net_signal < 0:
                mode = TrendMode.BEAR
                direction = -1
            else:
                mode = TrendMode.NEUTRAL
                direction = 0
        elif current_adx >= self.adx_threshold_weak:
            if net_signal > 0:
                mode = TrendMode.BULL
                direction = 1
            elif net_signal < 0:
                mode = TrendMode.BEAR
                direction = -1
            else:
                mode = TrendMode.NEUTRAL
                direction = 0
        else:
            mode = TrendMode.NEUTRAL
            direction = 0
        
        reason_parts = []
        reason_parts.append(f"ADX={current_adx:.1f}")
        reason_parts.append(f"ROC20={roc_20:.1f}%")
        reason_parts.append(f"MA{'↑' if ma_alignment > 0 else '↓' if ma_alignment < 0 else '→'}")
        reason_parts.append(f"MACD{'↑' if macd_bullish else '↓'}")
        reason_parts.append(f"Vol={vol_accel:.1f}x")
        
        return TrendSignal(
            mode=mode,
            strength=trend_strength,
            direction=direction,
            adx=current_adx,
            roc_20=roc_20,
            ma_alignment=ma_alignment,
            volume_trend=vol_accel,
            reason=", ".join(reason_parts)
        )
    
    def get_trend_position_bias(self, trend: TrendSignal) -> Tuple[float, str]:
        """
        Get position bias based on trend.
        
        Returns:
            Tuple of (bias_multiplier, reason)
            bias > 1: Increase long position
            bias < 1: Decrease position
            bias = 0: No position
        """
        biases = {
            TrendMode.STRONG_BULL: (1.5, "Strong bull - max long bias"),
            TrendMode.BULL: (1.2, "Bull trend - lean long"),
            TrendMode.NEUTRAL: (1.0, "Neutral - use base signals"),
            TrendMode.BEAR: (0.5, "Bear trend - reduce exposure"),
            TrendMode.STRONG_BEAR: (0.2, "Strong bear - minimal exposure"),
        }
        
        bias, reason = biases.get(trend.mode, (1.0, "Unknown mode"))
        return bias, reason


class MomentumFeatureGenerator:
    """
    Generate momentum features for enhanced signal detection.
    
    Phase 4: Add ROC, MACD, volume acceleration to feature set.
    """
    
    def __init__(self):
        self.roc_periods = [5, 10, 20, 50]
        self.volume_periods = [5, 10, 20]
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional momentum features
        """
        result = df.copy()
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        
        # Rate of Change
        for period in self.roc_periods:
            result[f'roc_{period}'] = (close / close.shift(period) - 1) * 100
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        result['macd'] = ema12 - ema26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # Volume acceleration
        for period in self.volume_periods:
            avg_vol = volume.rolling(period).mean()
            result[f'volume_accel_{period}'] = volume / avg_vol
        
        # Momentum oscillators
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        result['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        result['stoch_d'] = result['stoch_k'].rolling(3).mean()
        
        # Bollinger Band position
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        result['bb_position'] = (close - ma_20) / (2 * std_20 + 1e-10)
        
        # Average True Range (volatility)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr_14'] = tr.rolling(14).mean()
        result['atr_pct'] = result['atr_14'] / close * 100
        
        # Trend strength (ADX)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        result['adx'] = dx.ewm(span=14, adjust=False).mean()
        
        # Price relative to moving averages
        for ma_period in [20, 50, 200]:
            ma = close.rolling(ma_period).mean()
            result[f'price_to_ma_{ma_period}'] = (close / ma - 1) * 100
        
        # Moving average alignment
        ma_50 = close.rolling(50).mean()
        ma_200 = close.rolling(200).mean()
        result['ma_alignment'] = (ma_50 / ma_200 - 1) * 100
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of all generated feature names."""
        features = []
        
        # ROC
        for period in self.roc_periods:
            features.append(f'roc_{period}')
        
        # MACD
        features.extend(['macd', 'macd_signal', 'macd_histogram'])
        
        # Volume
        for period in self.volume_periods:
            features.append(f'volume_accel_{period}')
        
        # Oscillators
        features.extend(['rsi_14', 'stoch_k', 'stoch_d', 'bb_position'])
        
        # ATR
        features.extend(['atr_14', 'atr_pct'])
        
        # Trend
        features.append('adx')
        
        # MA position
        for ma_period in [20, 50, 200]:
            features.append(f'price_to_ma_{ma_period}')
        features.append('ma_alignment')
        
        return features


class DualModeStrategy:
    """
    Dual-mode strategy: Trend-following OR Mean-reversion.
    
    When trend is strong (ADX > 25), use trend-following.
    When no trend (ADX < 20), use mean-reversion.
    """
    
    def __init__(
        self,
        trend_adx_threshold: float = 25,
        mr_adx_threshold: float = 20
    ):
        self.trend_overlay = TrendFollowingOverlay()
        self.momentum_generator = MomentumFeatureGenerator()
        self.trend_adx_threshold = trend_adx_threshold
        self.mr_adx_threshold = mr_adx_threshold
    
    def get_strategy_mode(self, df: pd.DataFrame) -> Tuple[str, TrendSignal]:
        """
        Determine which strategy mode to use.
        
        Args:
            df: OHLCV data
            
        Returns:
            Tuple of (mode_string, trend_signal)
        """
        trend = self.trend_overlay.analyze_trend(df)
        
        if trend.adx >= self.trend_adx_threshold:
            return "trend_following", trend
        elif trend.adx <= self.mr_adx_threshold:
            return "mean_reversion", trend
        else:
            return "hybrid", trend
    
    def combine_signals(
        self,
        tda_signal: float,  # TDA/NN signal (-1 to 1)
        df: pd.DataFrame
    ) -> Tuple[float, str]:
        """
        Combine TDA signal with trend overlay.
        
        Args:
            tda_signal: Base TDA/NN signal
            df: OHLCV data for trend analysis
            
        Returns:
            Tuple of (combined_signal, reason)
        """
        mode, trend = self.get_strategy_mode(df)
        
        if mode == "trend_following":
            # In trend mode, align with trend direction
            if trend.direction > 0:
                # Bullish trend - boost long signals, dampen short
                if tda_signal > 0:
                    combined = min(1.0, tda_signal * 1.5)
                else:
                    combined = tda_signal * 0.3  # Dampen counter-trend
            elif trend.direction < 0:
                # Bearish trend - boost short signals
                if tda_signal < 0:
                    combined = max(-1.0, tda_signal * 1.5)
                else:
                    combined = tda_signal * 0.3
            else:
                combined = tda_signal
            
            reason = f"TREND mode: TDA {tda_signal:.2f} → {combined:.2f} ({trend.reason})"
            
        elif mode == "mean_reversion":
            # In mean reversion mode, use TDA signals directly
            combined = tda_signal
            reason = f"MR mode: Using TDA signal {tda_signal:.2f} ({trend.reason})"
            
        else:  # hybrid
            # Blend signals
            trend_bias, _ = self.trend_overlay.get_trend_position_bias(trend)
            combined = tda_signal * ((trend_bias + 1) / 2)  # Normalize bias
            reason = f"HYBRID mode: TDA {tda_signal:.2f} × bias {trend_bias:.2f} = {combined:.2f}"
        
        return combined, reason


if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Trend Following Overlay")
    print("=" * 60)
    
    # Download sample data
    spy = yf.download("SPY", start="2023-01-01", end="2024-01-01", progress=False)
    
    if len(spy) > 0:
        # Test trend analysis
        overlay = TrendFollowingOverlay()
        trend = overlay.analyze_trend(spy)
        
        print(f"\nTrend Analysis for SPY:")
        print(f"  Mode: {trend.mode.value}")
        print(f"  Strength: {trend.strength:.2f}")
        print(f"  Direction: {trend.direction}")
        print(f"  ADX: {trend.adx:.1f}")
        print(f"  ROC(20): {trend.roc_20:.2f}%")
        print(f"  MA Alignment: {trend.ma_alignment}")
        print(f"  Volume Trend: {trend.volume_trend:.2f}x")
        print(f"  Analysis: {trend.reason}")
        
        # Test momentum features
        print("\n" + "=" * 60)
        print("Testing Momentum Feature Generator")
        print("=" * 60)
        
        momentum_gen = MomentumFeatureGenerator()
        df_features = momentum_gen.generate_features(spy)
        
        print(f"\nGenerated {len(momentum_gen.get_feature_names())} momentum features:")
        for feat in momentum_gen.get_feature_names():
            if feat in df_features.columns:
                val = df_features[feat].iloc[-1]
                print(f"  {feat}: {val:.2f}")
        
        # Test dual mode strategy
        print("\n" + "=" * 60)
        print("Testing Dual Mode Strategy")
        print("=" * 60)
        
        dual_mode = DualModeStrategy()
        mode, trend = dual_mode.get_strategy_mode(spy)
        print(f"\nStrategy Mode: {mode}")
        print(f"Trend: {trend.mode.value}, ADX: {trend.adx:.1f}")
        
        # Test signal combination
        for tda_signal in [-0.5, 0, 0.5, 0.8]:
            combined, reason = dual_mode.combine_signals(tda_signal, spy)
            print(f"\nTDA signal {tda_signal:+.2f}: {reason}")
    
    print("\nTrend Following tests complete!")
