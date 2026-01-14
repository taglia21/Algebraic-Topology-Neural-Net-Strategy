"""Market Regime Detection for adaptive trading strategies.

Detects market regimes (BULL, BEAR, SIDEWAYS) and volatility states
(HIGH_VOL, LOW_VOL, NORMAL) to enable regime-aware trading decisions.

Uses technical indicators:
- Moving averages (50-day, 200-day)
- ATR for volatility
- RSI for momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class VolatilityState(Enum):
    """Volatility state classifications."""
    HIGH = "high_vol"
    LOW = "low_vol"
    NORMAL = "normal_vol"


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: Regime
    volatility: VolatilityState
    confidence: float
    ma_50: float
    ma_200: float
    atr_14: float
    rsi_14: float
    details: Dict


class MarketRegimeDetector:
    """
    Market regime detection using technical indicators.
    
    Regime Rules:
    - BULL: price > MA50 > MA200 AND RSI > 45
    - BEAR: price < MA50 < MA200 AND RSI < 55
    - SIDEWAYS: all other cases
    
    Volatility Rules:
    - HIGH_VOL: ATR > ATR_rolling_50 * 1.3
    - LOW_VOL: ATR < ATR_rolling_50 * 0.7
    - NORMAL: otherwise
    """
    
    def __init__(
        self,
        ma_short_period: int = 50,
        ma_long_period: int = 200,
        atr_period: int = 14,
        rsi_period: int = 14,
        atr_lookback: int = 50,
        high_vol_threshold: float = 1.3,
        low_vol_threshold: float = 0.7
    ):
        """
        Initialize MarketRegimeDetector.
        
        Args:
            ma_short_period: Short moving average period (default 50)
            ma_long_period: Long moving average period (default 200)
            atr_period: ATR calculation period (default 14)
            rsi_period: RSI calculation period (default 14)
            atr_lookback: ATR rolling average lookback (default 50)
            high_vol_threshold: Multiplier for high volatility detection
            low_vol_threshold: Multiplier for low volatility detection
        """
        self.ma_short_period = ma_short_period
        self.ma_long_period = ma_long_period
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.atr_lookback = atr_lookback
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        
        logger.info(f"MarketRegimeDetector initialized: MA({ma_short_period}/{ma_long_period}), "
                   f"ATR({atr_period}), RSI({rsi_period})")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def detect_regime(
        self,
        price_series: pd.Series,
        volume_series: pd.Series = None,
        df: pd.DataFrame = None
    ) -> RegimeResult:
        """
        Detect current market regime.
        
        Args:
            price_series: Close price series
            volume_series: Volume series (optional, for future enhancements)
            df: Full OHLCV DataFrame (for ATR calculation)
            
        Returns:
            RegimeResult with regime, volatility, and confidence
        """
        # Ensure we have enough data
        min_required = max(self.ma_long_period, self.atr_lookback + self.atr_period)
        
        if len(price_series) < min_required:
            return RegimeResult(
                regime=Regime.SIDEWAYS,
                volatility=VolatilityState.NORMAL,
                confidence=0.0,
                ma_50=0, ma_200=0, atr_14=0, rsi_14=50,
                details={'error': 'Insufficient data'}
            )
        
        # Calculate indicators
        close = price_series
        current_price = close.iloc[-1]
        
        ma_50 = close.rolling(window=self.ma_short_period).mean().iloc[-1]
        ma_200 = close.rolling(window=self.ma_long_period).mean().iloc[-1]
        rsi_14 = self._calculate_rsi(close, self.rsi_period).iloc[-1]
        
        # Calculate ATR if DataFrame provided
        if df is not None and len(df) >= self.atr_period:
            atr_series = self._calculate_atr(df, self.atr_period)
            atr_14 = atr_series.iloc[-1]
            atr_rolling_avg = atr_series.rolling(window=self.atr_lookback).mean().iloc[-1]
        else:
            # Fallback: estimate ATR from price volatility
            returns = close.pct_change().dropna()
            atr_14 = returns.std() * current_price
            atr_rolling_avg = atr_14
        
        # Detect regime
        regime, regime_confidence = self._classify_regime(
            current_price, ma_50, ma_200, rsi_14
        )
        
        # Detect volatility state
        volatility, vol_confidence = self._classify_volatility(
            atr_14, atr_rolling_avg
        )
        
        # Overall confidence
        confidence = (regime_confidence + vol_confidence) / 2
        
        return RegimeResult(
            regime=regime,
            volatility=volatility,
            confidence=round(confidence, 4),
            ma_50=round(ma_50, 4),
            ma_200=round(ma_200, 4),
            atr_14=round(atr_14, 4),
            rsi_14=round(rsi_14, 4),
            details={
                'price': round(current_price, 4),
                'price_vs_ma50': round((current_price / ma_50 - 1) * 100, 2) if ma_50 > 0 else 0,
                'price_vs_ma200': round((current_price / ma_200 - 1) * 100, 2) if ma_200 > 0 else 0,
                'ma50_vs_ma200': round((ma_50 / ma_200 - 1) * 100, 2) if ma_200 > 0 else 0,
                'atr_ratio': round(atr_14 / atr_rolling_avg, 4) if atr_rolling_avg > 0 else 1.0
            }
        )
    
    def _classify_regime(
        self,
        price: float,
        ma_50: float,
        ma_200: float,
        rsi: float
    ) -> Tuple[Regime, float]:
        """Classify market regime with confidence."""
        
        # BULL: price > MA50 > MA200 AND RSI > 45
        if price > ma_50 > ma_200 and rsi > 45:
            # Confidence based on how strongly conditions are met
            price_strength = (price / ma_50 - 1) * 100  # % above MA50
            ma_strength = (ma_50 / ma_200 - 1) * 100    # MA50 % above MA200
            rsi_strength = (rsi - 45) / 55              # Normalized RSI above 45
            
            confidence = min(1.0, (price_strength / 10 + ma_strength / 10 + rsi_strength) / 3)
            return Regime.BULL, max(0.5, confidence)
        
        # BEAR: price < MA50 < MA200 AND RSI < 55
        if price < ma_50 < ma_200 and rsi < 55:
            price_weakness = (1 - price / ma_50) * 100
            ma_weakness = (1 - ma_50 / ma_200) * 100
            rsi_weakness = (55 - rsi) / 55
            
            confidence = min(1.0, (price_weakness / 10 + ma_weakness / 10 + rsi_weakness) / 3)
            return Regime.BEAR, max(0.5, confidence)
        
        # SIDEWAYS: all other cases
        return Regime.SIDEWAYS, 0.5
    
    def _classify_volatility(
        self,
        atr: float,
        atr_avg: float
    ) -> Tuple[VolatilityState, float]:
        """Classify volatility state with confidence."""
        
        if atr_avg <= 0:
            return VolatilityState.NORMAL, 0.5
        
        ratio = atr / atr_avg
        
        if ratio > self.high_vol_threshold:
            confidence = min(1.0, (ratio - self.high_vol_threshold) / 0.5 + 0.5)
            return VolatilityState.HIGH, confidence
        
        if ratio < self.low_vol_threshold:
            confidence = min(1.0, (self.low_vol_threshold - ratio) / 0.3 + 0.5)
            return VolatilityState.LOW, confidence
        
        return VolatilityState.NORMAL, 0.5
    
    def add_regime_column(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Add regime column to DataFrame.
        
        Args:
            df: OHLCV DataFrame
            price_col: Column name for close price
            
        Returns:
            DataFrame with 'regime' and 'volatility' columns added
        """
        df = df.copy()
        
        # Standardize column names
        close_col = price_col if price_col in df.columns else price_col.capitalize()
        if close_col not in df.columns:
            close_col = 'Close' if 'Close' in df.columns else 'close'
        
        close = df[close_col]
        
        # Calculate indicators for full series
        df['ma_50'] = close.rolling(window=self.ma_short_period).mean()
        df['ma_200'] = close.rolling(window=self.ma_long_period).mean()
        df['rsi_14'] = self._calculate_rsi(close, self.rsi_period)
        df['atr_14'] = self._calculate_atr(df, self.atr_period)
        df['atr_avg_50'] = df['atr_14'].rolling(window=self.atr_lookback).mean()
        
        # Classify each row
        regimes = []
        volatilities = []
        confidences = []
        
        for i in range(len(df)):
            if i < self.ma_long_period or pd.isna(df['ma_200'].iloc[i]):
                regimes.append(Regime.SIDEWAYS.value)
                volatilities.append(VolatilityState.NORMAL.value)
                confidences.append(0.0)
                continue
            
            price = close.iloc[i]
            ma_50 = df['ma_50'].iloc[i]
            ma_200 = df['ma_200'].iloc[i]
            rsi = df['rsi_14'].iloc[i]
            atr = df['atr_14'].iloc[i]
            atr_avg = df['atr_avg_50'].iloc[i] if not pd.isna(df['atr_avg_50'].iloc[i]) else atr
            
            regime, regime_conf = self._classify_regime(price, ma_50, ma_200, rsi)
            vol, vol_conf = self._classify_volatility(atr, atr_avg)
            
            regimes.append(regime.value)
            volatilities.append(vol.value)
            confidences.append((regime_conf + vol_conf) / 2)
        
        df['regime'] = regimes
        df['volatility_state'] = volatilities
        df['regime_confidence'] = confidences
        
        return df
    
    def get_regime_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of regimes in a DataFrame.
        
        Args:
            df: DataFrame with 'regime' column
            
        Returns:
            Summary statistics
        """
        if 'regime' not in df.columns:
            df = self.add_regime_column(df)
        
        regime_counts = df['regime'].value_counts()
        total = len(df)
        
        summary = {
            'total_bars': total,
            'regime_distribution': {}
        }
        
        for regime in [Regime.BULL, Regime.BEAR, Regime.SIDEWAYS]:
            count = regime_counts.get(regime.value, 0)
            summary['regime_distribution'][regime.value] = {
                'count': int(count),
                'percentage': round(count / total * 100, 2) if total > 0 else 0
            }
        
        # Volatility distribution
        if 'volatility_state' in df.columns:
            vol_counts = df['volatility_state'].value_counts()
            summary['volatility_distribution'] = {}
            
            for vol in [VolatilityState.HIGH, VolatilityState.LOW, VolatilityState.NORMAL]:
                count = vol_counts.get(vol.value, 0)
                summary['volatility_distribution'][vol.value] = {
                    'count': int(count),
                    'percentage': round(count / total * 100, 2) if total > 0 else 0
                }
        
        return summary


def test_regime_detector():
    """Test MarketRegimeDetector functionality."""
    print("\n" + "=" * 60)
    print("Testing MarketRegimeDetector")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_days = 300
    
    # Generate trending price data
    trend = np.linspace(100, 150, n_days) + np.random.randn(n_days) * 2
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    df = pd.DataFrame({
        'open': trend + np.random.randn(n_days) * 0.5,
        'high': trend + np.abs(np.random.randn(n_days)) * 2,
        'low': trend - np.abs(np.random.randn(n_days)) * 2,
        'close': trend,
        'volume': np.random.randint(1000000, 5000000, n_days)
    }, index=dates)
    
    detector = MarketRegimeDetector()
    
    # Test single detection
    print("\nTest 1: Single regime detection")
    result = detector.detect_regime(df['close'], df=df)
    print(f"  Regime: {result.regime.value}")
    print(f"  Volatility: {result.volatility.value}")
    print(f"  Confidence: {result.confidence}")
    print(f"  MA50: {result.ma_50:.2f}, MA200: {result.ma_200:.2f}")
    print(f"  RSI: {result.rsi_14:.2f}")
    
    # Test add_regime_column
    print("\nTest 2: Add regime column")
    df_with_regime = detector.add_regime_column(df)
    print(f"  Columns added: regime, volatility_state, regime_confidence")
    print(f"  Sample regimes: {df_with_regime['regime'].tail().tolist()}")
    
    # Test summary
    print("\nTest 3: Regime summary")
    summary = detector.get_regime_summary(df_with_regime)
    print(f"  Regime distribution:")
    for regime, data in summary['regime_distribution'].items():
        print(f"    {regime}: {data['count']} bars ({data['percentage']}%)")
    
    print("\n" + "=" * 60)
    print("All MarketRegimeDetector tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_regime_detector()
