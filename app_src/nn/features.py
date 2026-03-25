"""
nn/features.py
==============
Feature engineering pipeline for the neural network module.

Assembles a complete feature matrix from price/volume data, technical
indicators, TDA-derived features, and cross-sectional signals.  All
computations are strictly causal (no future data leakage).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NNFeatureEngine:
    """Build the full feature matrix consumed by LSTM / Attention-LSTM models.

    Parameters
    ----------
    return_windows : list[int]
        Windows for multi-timeframe returns (default [1, 5, 10, 21, 63]).
    vol_windows : list[int]
        Windows for realised-volatility features (default [5, 10, 21]).
    rsi_period : int
        RSI lookback period (default 14).
    macd_fast : int
        MACD fast EMA period (default 12).
    macd_slow : int
        MACD slow EMA period (default 26).
    macd_signal : int
        MACD signal line period (default 9).
    bb_period : int
        Bollinger Band lookback (default 20).
    bb_std : float
        Bollinger Band standard deviations (default 2.0).
    atr_period : int
        ATR lookback (default 14).
    roc_period : int
        Rate-of-change lookback (default 10).
    volume_sma : int
        Volume SMA window for relative volume (default 20).
    """

    def __init__(
        self,
        return_windows: Optional[List[int]] = None,
        vol_windows: Optional[List[int]] = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        roc_period: int = 10,
        volume_sma: int = 20,
    ) -> None:
        self.return_windows = return_windows or [1, 5, 10, 21, 63]
        self.vol_windows = vol_windows or [5, 10, 21]
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.roc_period = roc_period
        self.volume_sma = volume_sma

        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(
        self,
        price_df: pd.DataFrame,
        volume_df: Optional[pd.DataFrame] = None,
        tda_features_df: Optional[pd.DataFrame] = None,
        sector_returns_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Assemble the full feature matrix.

        Parameters
        ----------
        price_df : pd.DataFrame
            Close prices with DatetimeIndex; single column or multi-column.
        volume_df : pd.DataFrame, optional
            Volume data aligned with price_df.
        tda_features_df : pd.DataFrame, optional
            Pre-computed TDA features from ``TDAFeatureExtractor.extract()``.
        sector_returns_df : pd.DataFrame, optional
            Sector-level returns for cross-sectional features.

        Returns
        -------
        pd.DataFrame
            Feature matrix with DatetimeIndex.  NaNs from rolling warm-up
            are forward-filled then any remaining NaN rows are dropped.
        """
        # Ensure we work with a single-column Series for price
        if isinstance(price_df, pd.DataFrame) and price_df.shape[1] == 1:
            price = price_df.iloc[:, 0]
        elif isinstance(price_df, pd.Series):
            price = price_df
        else:
            # Multi-asset: use first column as primary
            price = price_df.iloc[:, 0]

        volume: Optional[pd.Series] = None
        if volume_df is not None:
            if isinstance(volume_df, pd.DataFrame) and volume_df.shape[1] >= 1:
                volume = volume_df.iloc[:, 0]
            elif isinstance(volume_df, pd.Series):
                volume = volume_df

        parts: List[pd.DataFrame] = []

        # --- Price / volume features ---
        parts.append(self._price_features(price))
        if volume is not None:
            parts.append(self._volume_features(price, volume))

        # --- Technical features ---
        parts.append(self._technical_features(price, volume))

        # --- TDA features ---
        if tda_features_df is not None:
            parts.append(self._tda_features(tda_features_df))

        # --- Cross-sectional features ---
        if sector_returns_df is not None:
            parts.append(
                self._cross_sectional_features(price, sector_returns_df)
            )

        result = pd.concat(parts, axis=1)

        # Handle NaNs: forward fill then drop remaining
        result = result.ffill()
        result = result.dropna()

        self.feature_names = list(result.columns)
        return result

    def get_feature_groups(self) -> Dict[str, str]:
        """Return mapping of feature name → group label.

        Returns
        -------
        dict
            {feature_name: group} for every feature produced by the last
            ``build_features`` call.
        """
        groups: Dict[str, str] = {}
        for name in self.feature_names:
            if name.startswith(("ret_", "vol_", "log_ret")):
                groups[name] = "price"
            elif name.startswith(("rel_volume", "vwap")):
                groups[name] = "volume"
            elif name.startswith((
                "rsi", "macd", "bb_", "atr", "obv", "roc",
            )):
                groups[name] = "technical"
            elif name.startswith((
                "beta_", "persistence", "wasserstein",
                "spectral", "regime", "diffusion", "sci",
            )):
                groups[name] = "tda"
            elif name.startswith(("sector_", "breadth_", "avg_corr")):
                groups[name] = "cross_sectional"
            else:
                groups[name] = "other"
        return groups

    # ------------------------------------------------------------------
    # Feature groups (private)
    # ------------------------------------------------------------------

    def _price_features(self, price: pd.Series) -> pd.DataFrame:
        """Multi-timeframe returns, realised vol, log returns."""
        feats: Dict[str, pd.Series] = {}

        # Log returns
        feats["log_ret"] = np.log(price / price.shift(1))

        # Multi-timeframe returns
        for w in self.return_windows:
            feats[f"ret_{w}d"] = price.pct_change(w)

        # Realised volatility
        log_ret = feats["log_ret"]
        for w in self.vol_windows:
            feats[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

        return pd.DataFrame(feats, index=price.index)

    def _volume_features(
        self, price: pd.Series, volume: pd.Series,
    ) -> pd.DataFrame:
        """Relative volume and VWAP deviation."""
        feats: Dict[str, pd.Series] = {}

        vol_sma = volume.rolling(self.volume_sma).mean()
        feats["rel_volume"] = volume / vol_sma.replace(0, np.nan)

        # VWAP deviation (approx: typical price * volume / cumulative volume)
        typical_price = price  # using close as proxy
        cumvol = volume.cumsum()
        cumvol = cumvol.replace(0, np.nan)
        vwap = (typical_price * volume).cumsum() / cumvol
        feats["vwap_deviation"] = (price - vwap) / vwap.replace(0, np.nan)

        return pd.DataFrame(feats, index=price.index)

    def _technical_features(
        self,
        price: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """RSI, MACD, Bollinger, ATR, OBV, ROC."""
        feats: Dict[str, pd.Series] = {}

        # RSI
        feats["rsi"] = self._rsi(price, self.rsi_period)

        # MACD
        macd_line, signal_line, histogram = self._macd(
            price, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        feats["macd_line"] = macd_line
        feats["macd_signal"] = signal_line
        feats["macd_hist"] = histogram

        # Bollinger Band width
        sma = price.rolling(self.bb_period).mean()
        std = price.rolling(self.bb_period).std()
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        feats["bb_width"] = (upper - lower) / sma.replace(0, np.nan)

        # ATR (using close-based proxy when no OHLC available)
        feats["atr"] = self._atr_close(price, self.atr_period)

        # OBV
        if volume is not None:
            feats["obv"] = self._obv(price, volume)

        # Rate of Change
        feats["roc"] = price.pct_change(self.roc_period)

        return pd.DataFrame(feats, index=price.index)

    def _tda_features(self, tda_df: pd.DataFrame) -> pd.DataFrame:
        """Pass through TDA features with consistent naming."""
        expected_cols = [
            "beta_0", "beta_1", "persistence_entropy", "wasserstein_dist",
            "spectral_gap", "regime", "diffusion_residual_mean",
            "diffusion_residual_std", "sci",
        ]
        available = [c for c in expected_cols if c in tda_df.columns]
        return tda_df[available].copy()

    def _cross_sectional_features(
        self,
        price: pd.Series,
        sector_returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sector relative strength, breadth, average correlation."""
        feats: Dict[str, pd.Series] = {}

        stock_ret = price.pct_change()

        # Sector relative strength
        if sector_returns_df.shape[1] > 0:
            sector_mean = sector_returns_df.mean(axis=1)
            feats["sector_rel_strength"] = stock_ret - sector_mean

        # Breadth: % of sector members above 50d SMA
        if sector_returns_df.shape[1] > 0:
            cum_ret = (1 + sector_returns_df).cumprod()
            sma_50 = cum_ret.rolling(50).mean()
            above = (cum_ret > sma_50).sum(axis=1)
            total = sector_returns_df.shape[1]
            feats["breadth_pct_above_50sma"] = above / total

        # Average pairwise correlation (rolling 60d)
        if sector_returns_df.shape[1] > 1:
            feats["avg_corr"] = sector_returns_df.rolling(60).corr().groupby(
                level=0,
            ).apply(
                lambda c: c.values[np.triu_indices_from(c.values, k=1)].mean()
                if c.shape[0] > 1 else 0.0,
            )

        return pd.DataFrame(feats, index=price.index)

    # ------------------------------------------------------------------
    # Technical indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(price: pd.Series, period: int) -> pd.Series:
        """Compute RSI using exponential moving average of gains/losses."""
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _macd(
        price: pd.Series, fast: int, slow: int, signal: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD line, signal line, and histogram."""
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _atr_close(price: pd.Series, period: int) -> pd.Series:
        """ATR approximation using close-only data (abs daily range)."""
        high_low_proxy = price.diff().abs()
        return high_low_proxy.rolling(period).mean()

    @staticmethod
    def _obv(price: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(price.diff())
        obv = (direction * volume).cumsum()
        return obv
