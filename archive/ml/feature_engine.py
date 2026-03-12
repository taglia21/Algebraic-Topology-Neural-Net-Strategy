"""
ml/feature_engine.py
====================
Feature engineering pipeline for the ATNN ML trading system.

Computes ~150 features across seven categories using fully vectorised
pandas/numpy operations — no row-level Python loops.

Categories
----------
1. Price-Based         (~25 features)
2. Volume-Based        (~15 features)
3. Volatility          (~20 features)
4. Cross-Sectional     (~25 features)
5. Macro / Sentiment   (~25 features)  — placeholder columns when data absent
6. Engineered          (~25 features)
7. Microstructure      (~10 features)  — placeholder columns when data absent

Usage
-----
    from ml.feature_engine import FeatureEngine

    engine = FeatureEngine()
    features = engine.compute_features(price_data, symbol="AAPL")
    print(engine.get_feature_names())
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252

# Minimum number of valid rows required before returning any features.
# (longest lookback is 200-day SMA)
MIN_ROWS: int = 201


# ===========================================================================
# Helper: pure-numpy / pandas implementations (no TA-Lib)
# ===========================================================================

def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (α = 1/period)."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index via Wilder smoothing.

    Parameters
    ----------
    close:
        Closing-price series.
    period:
        RSI look-back window.

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_ema(gain, period)
    avg_loss = _wilder_ema(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard exponential moving average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _wilder_ema(tr, period)


def _bollinger_pctb(close: pd.Series, window: int = 20,
                    n_std: float = 2.0) -> pd.Series:
    """Bollinger %B = (price - lower) / (upper - lower)."""
    sma = _sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / band_width


def _keltner_position(close: pd.Series, high: pd.Series,
                      low: pd.Series, period: int = 20,
                      mult: float = 2.0) -> pd.Series:
    """Position of close within Keltner Channel: (close - lower) / (upper - lower)."""
    mid = _ema(close, span=period)
    atr = _atr(high, low, close, period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    band = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / band


def _hurst_exponent(series: pd.Series, min_lag: int = 2,
                    max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent via simplified R/S analysis (vectorised).

    Uses a 60-day rolling window. Values < 0.5 indicate mean-reversion,
    0.5 = random walk, > 0.5 indicates trending / persistence.

    This implementation uses a fast approximation: it computes R/S at two
    lag scales (short=5, long=15) and estimates H from the log-log slope
    of those two points. This avoids the expensive per-bar Python callback
    while retaining the signal's economic interpretation.

    Parameters
    ----------
    series:
        Price series.
    min_lag, max_lag:
        Ignored (kept for API compatibility). Uses lags 5 and 15 internally.

    Returns
    -------
    pd.Series
        Rolling Hurst exponent (window = 60).
    """
    window = 60
    log_prices = np.log(series.clip(lower=1e-9))
    log_returns = log_prices.diff()

    def _rolling_rs(returns: pd.Series, lag: int) -> pd.Series:
        """Rescaled range for a given lag, computed via rolling operations."""
        # Rolling mean deviation
        roll_mean = returns.rolling(lag, min_periods=lag).mean()
        deviation = returns - roll_mean
        # Cumulative deviation within each window (approximate via rolling sum)
        cum_dev = deviation.rolling(lag, min_periods=lag).sum()
        # Range approximation: use rolling max - rolling min of cumulative deviations
        roll_range = (
            deviation.rolling(lag, min_periods=lag).max()
            - deviation.rolling(lag, min_periods=lag).min()
        )
        # Standard deviation
        roll_std = returns.rolling(lag, min_periods=lag).std()
        rs = roll_range / roll_std.replace(0.0, np.nan)
        return rs

    # Compute R/S at two scales within the rolling window
    lag_short = 5
    lag_long = 15
    rs_short = _rolling_rs(log_returns, lag_short)
    rs_long = _rolling_rs(log_returns, lag_long)

    # Ensure both are within the broader 60-day window
    rs_short = rs_short.rolling(window, min_periods=window).mean()
    rs_long = rs_long.rolling(window, min_periods=window).mean()

    # Hurst = log(RS_long/RS_short) / log(lag_long/lag_short)
    log_rs_ratio = np.log(rs_long / rs_short.replace(0.0, np.nan))
    log_lag_ratio = np.log(lag_long / lag_short)

    hurst = log_rs_ratio / log_lag_ratio
    # Clip to valid Hurst range [0, 1]
    return hurst.clip(0.0, 1.0)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index (0–100)."""
    typical = (high + low + close) / 3.0
    raw_mf = typical * volume
    direction = typical.diff()
    pos_mf = raw_mf.where(direction > 0, 0.0)
    neg_mf = raw_mf.where(direction < 0, 0.0)

    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum()

    mfr = pos_sum / neg_sum.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + mfr)


def _chaikin_mf(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0.0, np.nan)
    mf_vol = clv * volume
    return (
        mf_vol.rolling(period, min_periods=period).sum()
        / volume.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    )


def _ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series) -> pd.Series:
    """Accumulation / Distribution line."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0.0, np.nan)
    return (clv * volume).cumsum()


def _garman_klass_vol(open_: pd.Series, high: pd.Series,
                      low: pd.Series, close: pd.Series,
                      window: int = 20) -> pd.Series:
    """Garman-Klass volatility estimator (annualised)."""
    log_hl = np.log(high / low.replace(0.0, np.nan)) ** 2
    log_co = np.log(close / open_.replace(0.0, np.nan)) ** 2
    gk = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return (gk.rolling(window=window, min_periods=window).mean() * TRADING_DAYS_PER_YEAR) ** 0.5


def _parkinson_vol(high: pd.Series, low: pd.Series,
                   window: int = 20) -> pd.Series:
    """Parkinson volatility estimator (annualised)."""
    log_hl2 = np.log(high / low.replace(0.0, np.nan)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return (
        (factor * log_hl2).rolling(window=window, min_periods=window).mean()
        * TRADING_DAYS_PER_YEAR
    ) ** 0.5


def _fracdiff(series: pd.Series, d: float = 0.4,
              threshold: float = 1e-5) -> pd.Series:
    """Fractionally differentiated series (Lopez de Prado Chapter 5).

    Computes the fractionally differentiated series using the fixed-width
    window method. Weights are computed as:
        w_k = prod_{j=0}^{k-1} (d - j) / (j + 1)

    Parameters
    ----------
    series:
        Price series (must be positive).
    d:
        Differencing order, typically 0.3–0.5 to achieve stationarity while
        preserving memory.
    threshold:
        Truncate weight vector when |w_k| < threshold.

    Returns
    -------
    pd.Series
        Fractionally differentiated series, NaN for insufficient history.
    """
    # Compute weights until |w| < threshold
    weights: List[float] = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights_arr = np.array(weights)
    w_len = len(weights_arr)

    log_series = np.log(series.clip(lower=1e-9))
    vals = log_series.values
    n = len(vals)

    # Vectorised convolution: dot product of reversed weights with rolling window
    if w_len > n:
        return pd.Series(np.full(n, np.nan), index=series.index, name="fracdiff_close")

    # Use numpy convolution for O(n) performance instead of O(n*w_len) Python loop
    conv = np.convolve(vals, weights_arr, mode='full')[:n]
    result = np.full(n, np.nan)
    result[w_len - 1:] = conv[w_len - 1:]

    return pd.Series(result, index=series.index, name="fracdiff_close")


def _shannon_entropy(returns: pd.Series, window: int = 20,
                     n_bins: int = 10) -> pd.Series:
    """Rolling Shannon entropy of discretised returns.

    Parameters
    ----------
    returns:
        Log or simple return series.
    window:
        Rolling window in bars.
    n_bins:
        Number of histogram bins.

    Returns
    -------
    pd.Series
        Shannon entropy (nats).
    """
    def _entropy_scalar(x: np.ndarray) -> float:
        counts, _ = np.histogram(x, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    return returns.rolling(window=window, min_periods=window).apply(
        _entropy_scalar, raw=True
    )


# ===========================================================================
# FeatureEngine
# ===========================================================================

class FeatureEngine:
    """Computes all features for the ML pipeline.

    All features are computed using vectorised pandas/numpy operations.
    NaN values are preserved wherever lookback requirements are not yet met.

    Parameters
    ----------
    spy_data:
        Optional SPY OHLCV DataFrame for cross-sectional features.
        If None, cross-sectional features are set to NaN.
    sector_data:
        Optional dict mapping sector ETF ticker → OHLCV DataFrame
        used for sector relative strength features.
    macro_data:
        Optional DataFrame with macro columns:
        ``vix``, ``vxv``, ``put_call_ratio``, ``yield_curve_slope``.
    """

    def __init__(
        self,
        spy_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.spy_data = spy_data
        self.sector_data = sector_data or {}
        self.macro_data = macro_data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(
        self,
        price_data: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute all features for a single symbol's OHLCV data.

        Parameters
        ----------
        price_data:
            DataFrame indexed by datetime with columns:
            ``open``, ``high``, ``low``, ``close``, ``volume``.
            Column names are case-insensitive.
        symbol:
            Optional ticker string, used only for logging.

        Returns
        -------
        pd.DataFrame
            Feature DataFrame indexed by the same datetime index as
            *price_data*.  Each column is a named feature.  NaN values
            appear where lookback requirements are not yet satisfied.

        Raises
        ------
        KeyError
            If required columns are missing from *price_data*.
        """
        df = price_data.copy()
        df.columns = [c.lower() for c in df.columns]

        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                if col == "open":
                    df["open"] = df["close"]   # synthetic open = close
                elif col == "volume":
                    df["volume"] = np.nan
                else:
                    raise KeyError(
                        f"price_data missing required column '{col}'. "
                        f"Available: {list(df.columns)}"
                    )

        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        log_returns = np.log(close / close.shift(1))

        features: Dict[str, pd.Series] = {}

        # ------------------------------------------------------------------
        # 1. Price-based features
        # ------------------------------------------------------------------
        features.update(
            self._price_features(open_, high, low, close, log_returns)
        )

        # ------------------------------------------------------------------
        # 2. Volume-based features
        # ------------------------------------------------------------------
        features.update(
            self._volume_features(high, low, close, volume, log_returns)
        )

        # ------------------------------------------------------------------
        # 3. Volatility features
        # ------------------------------------------------------------------
        features.update(
            self._volatility_features(open_, high, low, close, log_returns)
        )

        # ------------------------------------------------------------------
        # 4. Cross-sectional features
        # ------------------------------------------------------------------
        features.update(
            self._cross_sectional_features(close, log_returns, symbol)
        )

        # ------------------------------------------------------------------
        # 5. Macro / sentiment features
        # ------------------------------------------------------------------
        features.update(
            self._macro_features(df)
        )

        # ------------------------------------------------------------------
        # 6. Engineered features
        # ------------------------------------------------------------------
        features.update(
            self._engineered_features(close, log_returns)
        )

        # ------------------------------------------------------------------
        # 7. Microstructure placeholders
        # ------------------------------------------------------------------
        features.update(
            self._microstructure_features(df)
        )

        out = pd.DataFrame(features, index=df.index)

        logger.debug(
            f"FeatureEngine: computed {len(out.columns)} features "
            f"for {symbol or 'unknown'} over {len(out)} bars."
        )
        return out

    def get_feature_names(self) -> List[str]:
        """Return the canonical ordered list of feature column names.

        Returns
        -------
        List[str]
            All feature names produced by :meth:`compute_features`.
        """
        # Build a minimal dummy DataFrame to enumerate names
        n = MAX_LOOKBACK = 250
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        dummy = pd.DataFrame({
            "open":   np.linspace(100, 110, n),
            "high":   np.linspace(101, 111, n),
            "low":    np.linspace(99, 109, n),
            "close":  np.linspace(100, 110, n),
            "volume": np.ones(n) * 1_000_000,
        }, index=idx)
        try:
            feats = self.compute_features(dummy, symbol="_dummy")
            return list(feats.columns)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"get_feature_names: dummy run failed ({exc}); returning empty list.")
            return []

    # ------------------------------------------------------------------
    # Category helpers
    # ------------------------------------------------------------------

    def _price_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        log_returns: pd.Series,
    ) -> Dict[str, pd.Series]:
        """Price-based features (~25)."""
        f: Dict[str, pd.Series] = {}

        # RSI at multiple periods
        for period in (2, 7, 14):
            f[f"rsi_{period}"] = _rsi(close, period)

        # MACD (12, 26, 9)
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        histogram = macd_line - signal_line

        f["macd_line"] = macd_line
        f["macd_signal"] = signal_line
        f["macd_histogram"] = histogram
        # Zero-cross: +1 if macd_line crossed above zero, -1 below, 0 otherwise
        prev_macd = macd_line.shift(1)
        f["macd_zero_cross"] = np.where(
            (macd_line > 0) & (prev_macd <= 0), 1.0,
            np.where((macd_line < 0) & (prev_macd >= 0), -1.0, 0.0)
        )

        # Bollinger %B at periods 20 and 50
        f["bollinger_pctb_20"] = _bollinger_pctb(close, window=20)
        f["bollinger_pctb_50"] = _bollinger_pctb(close, window=50)

        # Rate of Change
        for period in (5, 20, 60):
            f[f"roc_{period}"] = (close / close.shift(period).replace(0.0, np.nan) - 1.0)

        # ATR ratio: ATR_14 / close
        atr14 = _atr(high, low, close, period=14)
        f["atr_ratio_14"] = atr14 / close.replace(0.0, np.nan)

        # Hurst exponent (rolling 60-day window on close prices)
        f["hurst_exp"] = _hurst_exponent(close)

        # Moving average crossovers: positive means fast > slow
        sma5   = _sma(close, 5)
        sma20  = _sma(close, 20)
        sma50  = _sma(close, 50)
        sma200 = _sma(close, 200)

        f["ma_cross_5_20"]   = (sma5 - sma20)  / close.replace(0.0, np.nan)
        f["ma_cross_20_50"]  = (sma20 - sma50)  / close.replace(0.0, np.nan)
        f["ma_cross_50_200"] = (sma50 - sma200) / close.replace(0.0, np.nan)

        # Price distance from SMA (as % of price)
        f["price_dist_sma20"]  = (close - sma20)  / close.replace(0.0, np.nan)
        f["price_dist_sma50"]  = (close - sma50)  / close.replace(0.0, np.nan)
        f["price_dist_sma200"] = (close - sma200) / close.replace(0.0, np.nan)

        # Keltner Channel position
        f["keltner_pos"] = _keltner_position(close, high, low)

        return f

    def _volume_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        log_returns: pd.Series,
    ) -> Dict[str, pd.Series]:
        """Volume-based features (~15)."""
        f: Dict[str, pd.Series] = {}

        vol_available = volume.notna().any()

        # OBV rate-of-change (stationary, no cumulative look-ahead bias)
        # Raw OBV is a cumulative sum and is non-stationary; its pct_change is
        # stationary and contains no future information.
        if vol_available:
            raw_obv = _obv(close, volume)
            f["obv_roc_20"] = raw_obv.pct_change(20)  # 20-bar rate of change
            f["obv_roc_5"]  = raw_obv.pct_change(5)   # 5-bar rate of change
        else:
            f["obv_roc_20"] = pd.Series(np.nan, index=close.index)
            f["obv_roc_5"]  = pd.Series(np.nan, index=close.index)

        # VWAP deviation: (close - VWAP) / VWAP
        # Approximate daily VWAP as rolling sum(tp * vol) / rolling sum(vol)
        if vol_available:
            tp = (high + low + close) / 3.0
            tp_vol = tp * volume
            vwap = (
                tp_vol.rolling(20, min_periods=1).sum()
                / volume.rolling(20, min_periods=1).sum().replace(0.0, np.nan)
            )
            f["vwap_deviation"] = (close - vwap) / vwap.replace(0.0, np.nan)
        else:
            f["vwap_deviation"] = pd.Series(np.nan, index=close.index)

        # Volume ratio: 5-day avg / 20-day avg
        if vol_available:
            vol_ma5  = volume.rolling(5,  min_periods=5).mean()
            vol_ma20 = volume.rolling(20, min_periods=20).mean()
            f["volume_ratio_5_20"] = vol_ma5 / vol_ma20.replace(0.0, np.nan)
        else:
            f["volume_ratio_5_20"] = pd.Series(np.nan, index=close.index)

        # Relative volume vs 20-day average
        if vol_available:
            f["relative_volume"] = volume / volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
        else:
            f["relative_volume"] = pd.Series(np.nan, index=close.index)

        # Money Flow Index
        if vol_available:
            f["mfi_14"] = _mfi(high, low, close, volume, 14)
        else:
            f["mfi_14"] = pd.Series(np.nan, index=close.index)

        # Chaikin Money Flow
        if vol_available:
            f["chaikin_mf_20"] = _chaikin_mf(high, low, close, volume, 20)
        else:
            f["chaikin_mf_20"] = pd.Series(np.nan, index=close.index)

        # Volume-weighted RSI
        if vol_available:
            weighted_close = (close * volume).rolling(14, min_periods=14).sum() / \
                             volume.rolling(14, min_periods=14).sum().replace(0.0, np.nan)
            f["vw_rsi_14"] = _rsi(weighted_close.ffill(), 14)
        else:
            f["vw_rsi_14"] = pd.Series(np.nan, index=close.index)

        # Chaikin Oscillator (3-day EMA of A/D minus 10-day EMA of A/D)
        # Replaces the raw cumulative A/D line which has look-ahead bias from
        # the full-dataset cumsum.  The Chaikin Oscillator is stationary and
        # captures momentum in the A/D line without leaking future data.
        if vol_available:
            raw_ad = _ad_line(high, low, close, volume)
            f["chaikin_osc"] = raw_ad.ewm(span=3, adjust=False).mean() - raw_ad.ewm(span=10, adjust=False).mean()
        else:
            f["chaikin_osc"] = pd.Series(np.nan, index=close.index)

        return f

    def _volatility_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        log_returns: pd.Series,
    ) -> Dict[str, pd.Series]:
        """Volatility features (~20)."""
        f: Dict[str, pd.Series] = {}

        ann = np.sqrt(TRADING_DAYS_PER_YEAR)

        # Realized volatility at multiple horizons (annualised)
        for window in (5, 20, 60):
            rv = log_returns.rolling(window=window, min_periods=window).std() * ann
            f[f"realized_vol_{window}d"] = rv

        # Vol compression signal: rv5 / rv60 (< 1 means compressed / coiling)
        f["vol_compression_5_60"] = (
            f["realized_vol_5d"] / f["realized_vol_60d"].replace(0.0, np.nan)
        )

        # ATR at 14 and 21 days
        f["atr_14"] = _atr(high, low, close, period=14)
        f["atr_21"] = _atr(high, low, close, period=21)

        # Garman-Klass estimator
        f["garman_klass_vol"] = _garman_klass_vol(open_, high, low, close)

        # Parkinson estimator
        f["parkinson_vol"] = _parkinson_vol(high, low)

        # Vol-of-vol: std of 20d rolling vol over 60-day window
        rv20 = log_returns.rolling(20, min_periods=20).std() * ann
        f["vol_of_vol"] = rv20.rolling(60, min_periods=60).std()

        # High-low range ratio (normalised)
        f["hl_range_ratio"] = (high - low) / close.replace(0.0, np.nan)

        # Skewness of returns (rolling 20d)
        f["return_skew_20d"] = log_returns.rolling(20, min_periods=20).skew()

        # Kurtosis of returns (rolling 20d)
        f["return_kurt_20d"] = log_returns.rolling(20, min_periods=20).kurt()

        # Up/down volatility asymmetry
        up_vol   = log_returns.clip(lower=0).rolling(20, min_periods=20).std() * ann
        down_vol = log_returns.clip(upper=0).rolling(20, min_periods=20).std() * ann
        f["vol_asymmetry"] = up_vol / down_vol.replace(0.0, np.nan)

        return f

    def _cross_sectional_features(
        self,
        close: pd.Series,
        log_returns: pd.Series,
        symbol: Optional[str],
    ) -> Dict[str, pd.Series]:
        """Cross-sectional features vs SPY (~25).

        If spy_data is not provided, these features are NaN placeholders.
        """
        f: Dict[str, pd.Series] = {}
        nan_series = pd.Series(np.nan, index=close.index)

        spy_close: Optional[pd.Series] = None
        if self.spy_data is not None:
            sd = self.spy_data.copy()
            sd.columns = [c.lower() for c in sd.columns]
            if "close" in sd.columns:
                spy_close = sd["close"].astype(float).reindex(close.index).ffill()

        if spy_close is not None:
            spy_returns = np.log(spy_close / spy_close.shift(1))

            # Beta to SPY (60-day rolling OLS)
            def _rolling_beta(stock_r: pd.Series, spy_r: pd.Series,
                              window: int = 60) -> pd.Series:
                cov = stock_r.rolling(window, min_periods=window).cov(spy_r)
                var = spy_r.rolling(window, min_periods=window).var()
                return cov / var.replace(0.0, np.nan)

            beta_60 = _rolling_beta(log_returns, spy_returns, 60)
            f["beta_spy_60d"] = beta_60

            # Correlation to SPY (60-day rolling)
            f["corr_spy_60d"] = log_returns.rolling(60, min_periods=60).corr(spy_returns)

            # Sector relative strength (stock return vs SPY return, 20d)
            stock_ret_20 = close / close.shift(20).replace(0.0, np.nan) - 1.0
            spy_ret_20   = spy_close / spy_close.shift(20).replace(0.0, np.nan) - 1.0
            f["rel_strength_spy_20d"] = stock_ret_20 - spy_ret_20

            # Beta-adjusted excess return (alpha proxy)
            f["beta_adj_excess_return"] = log_returns - beta_60 * spy_returns

            # Lagged cross-correlation with SPY (lag 0–3 days)
            for lag in range(0, 4):
                f[f"xcorr_spy_lag{lag}"] = (
                    log_returns.rolling(20, min_periods=20)
                    .corr(spy_returns.shift(lag))
                )

            # Industry momentum rank (approximated as 12-1 month return rank)
            ret_12_1 = (
                close / close.shift(252).replace(0.0, np.nan)
                - close / close.shift(21).replace(0.0, np.nan)
            )
            # Normalised to [-1, 1] range using rolling quantile
            f["momentum_12_1"] = ret_12_1
            f["momentum_1m"] = close / close.shift(21).replace(0.0, np.nan) - 1.0
            f["momentum_3m"] = close / close.shift(63).replace(0.0, np.nan) - 1.0
            f["momentum_6m"] = close / close.shift(126).replace(0.0, np.nan) - 1.0

        else:
            for name in [
                "beta_spy_60d", "corr_spy_60d", "rel_strength_spy_20d",
                "beta_adj_excess_return",
                "xcorr_spy_lag0", "xcorr_spy_lag1", "xcorr_spy_lag2",
                "xcorr_spy_lag3",
                "momentum_12_1", "momentum_1m", "momentum_3m", "momentum_6m",
            ]:
                f[name] = nan_series.copy()

        # Sector relative strength vs sector ETF
        sector_etf_return: Optional[pd.Series] = None
        if symbol and symbol in self.sector_data:
            sec_df = self.sector_data[symbol].copy()
            sec_df.columns = [c.lower() for c in sec_df.columns]
            if "close" in sec_df.columns:
                sec_close = sec_df["close"].astype(float).reindex(close.index).ffill()
                stock_ret_20 = close / close.shift(20).replace(0.0, np.nan) - 1.0
                sec_ret_20   = sec_close / sec_close.shift(20).replace(0.0, np.nan) - 1.0
                sector_etf_return = stock_ret_20 - sec_ret_20

        f["rel_strength_sector_20d"] = (
            sector_etf_return if sector_etf_return is not None
            else nan_series.copy()
        )

        # 52-week high/low distance
        high_52w = close.rolling(252, min_periods=60).max()
        low_52w  = close.rolling(252, min_periods=60).min()
        f["dist_52w_high"] = (close - high_52w) / high_52w.replace(0.0, np.nan)
        f["dist_52w_low"]  = (close - low_52w)  / low_52w.replace(0.0, np.nan)

        return f

    def _macro_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Macro / sentiment placeholder features (~25).

        Reads from self.macro_data if available, otherwise returns NaN
        series.  Strategies must handle NaN gracefully.
        """
        f: Dict[str, pd.Series] = {}
        nan_series = pd.Series(np.nan, index=df.index)

        macro: Optional[pd.DataFrame] = None
        if self.macro_data is not None:
            macro = self.macro_data.copy()
            macro.columns = [c.lower() for c in macro.columns]
            macro = macro.reindex(df.index).ffill()

        def _get_col(col: str) -> pd.Series:
            if macro is not None and col in macro.columns:
                return macro[col].astype(float)
            return nan_series.copy()

        # VIX features — only include if VIX data exists in macro_data or df
        vix = _get_col("vix")
        has_vix = macro is not None and "vix" in macro.columns
        if not has_vix and "vix" in df.columns:
            vix = df["vix"].astype(float)
            has_vix = True

        if has_vix and vix.notna().sum() > 20:
            f["vix_level"] = vix
            f["vix_change_1d"] = vix.diff(1)
            f["vix_change_5d"] = vix.diff(5)
            f["vix_pct_change_1d"] = vix / vix.shift(1).replace(0.0, np.nan) - 1.0
            f["vix_ma_ratio"] = vix / vix.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
            f["vix_percentile_252d"] = vix.rolling(252, min_periods=60).rank(pct=True)

            # VIX term structure (VIX/VXV)
            vxv = _get_col("vxv")
            if macro is not None and "vxv" in macro.columns:
                f["vix_term_structure"] = vix / vxv.replace(0.0, np.nan)

        # Put/Call ratio — only if available
        if macro is not None and "put_call_ratio" in macro.columns:
            f["put_call_ratio"] = _get_col("put_call_ratio")
            f["put_call_ma5"] = f["put_call_ratio"].rolling(5, min_periods=5).mean()

        # Yield curve slope — only if available
        if macro is not None and "yield_curve_slope" in macro.columns:
            yc = _get_col("yield_curve_slope")
            f["yield_curve_slope"] = yc
            f["yield_curve_slope_change_5d"] = yc.diff(5)
            f["yield_curve_slope_ma20"] = yc.rolling(20, min_periods=20).mean()

        # Credit spread — only if available
        if macro is not None and "credit_spread" in macro.columns:
            f["credit_spread"] = _get_col("credit_spread")
            f["credit_spread_change_5d"] = f["credit_spread"].diff(5)

        # TED spread — only include if macro_data provides it
        ted = _get_col("ted_spread")
        if macro is not None and "ted_spread" in macro.columns:
            f["ted_spread"] = ted

        # Dollar index — only include if macro_data provides it
        dxy = _get_col("dxy")
        if macro is not None and "dxy" in macro.columns:
            f["dxy_level"] = dxy
            f["dxy_change_5d"] = dxy.diff(5)

        # --- Below features ONLY included when macro_data provides them ---
        # (Removed as NaN placeholders — they add noise to ML training)
        for col_name in ("advance_decline_ratio", "market_breadth", "news_sentiment",
                         "short_interest_ratio", "earnings_surprise", "iv_rank",
                         "macro_regime"):
            val = _get_col(col_name)
            if macro is not None and col_name in macro.columns:
                f[col_name] = val

        return f

    def _engineered_features(
        self,
        close: pd.Series,
        log_returns: pd.Series,
    ) -> Dict[str, pd.Series]:
        """Engineered / statistical features (~25)."""
        f: Dict[str, pd.Series] = {}

        # Fractionally differentiated close prices (d ≈ 0.4)
        f["fracdiff_close"] = _fracdiff(close, d=0.4)

        # Shannon entropy of returns (rolling 20-day)
        f["entropy_returns_20d"] = _shannon_entropy(log_returns, window=20)

        # Autoregressive features: lag-1 through lag-5 returns
        for lag in range(1, 6):
            f[f"return_lag{lag}"] = log_returns.shift(lag)

        # Log returns (current)
        f["log_return"] = log_returns

        # Squared log returns (variance proxy)
        f["log_return_sq"] = log_returns ** 2

        # Rolling autocorrelation (lag-1, 20-day window)
        f["autocorr_lag1_20d"] = log_returns.rolling(20, min_periods=20).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else np.nan,
            raw=False,
        )

        # Return skewness (rolling 20d)
        f["return_skew_20d_eng"] = log_returns.rolling(20, min_periods=20).skew()

        # Return kurtosis (rolling 20d)
        f["return_kurt_20d_eng"] = log_returns.rolling(20, min_periods=20).kurt()

        # Cumulative return over various windows
        for window in (5, 10, 20):
            f[f"cum_return_{window}d"] = log_returns.rolling(window=window, min_periods=window).sum()

        # GARCH-proxy features
        # EWM variance (λ=0.94, similar to RiskMetrics)
        ewm_var = log_returns.ewm(span=20, min_periods=20, adjust=False).var()
        f["ewm_variance"] = ewm_var
        f["ewm_vol_ratio"] = log_returns ** 2 / ewm_var.replace(0.0, np.nan)

        # Realised covariance with lagged self (persistence)
        f["return_persistence"] = (log_returns * log_returns.shift(1)).rolling(20, min_periods=20).mean()

        # Deviation from rolling median (non-linear distance)
        med20 = log_returns.rolling(20, min_periods=20).median()
        f["return_median_dev"] = log_returns - med20

        # Cross-return momentum (3-5 day sum / vol)
        ret_5_sum = log_returns.rolling(5, min_periods=5).sum()
        vol_5 = log_returns.rolling(5, min_periods=5).std().replace(0.0, np.nan)
        f["momentum_score_5d"] = ret_5_sum / vol_5

        return f

    def _microstructure_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Microstructure placeholder features (~10).

        Returns NaN when intraday / L2 data not available.
        Strategies must handle missing gracefully.
        """
        f: Dict[str, pd.Series] = {}
        nan_series = pd.Series(np.nan, index=df.index)

        # Bid-ask spread proxy (uses H-L as a crude daily proxy when unavailable)
        if "bid_ask_spread" in df.columns:
            f["bid_ask_spread"] = df["bid_ask_spread"].astype(float)
        else:
            # Corwin-Schultz estimator approximation using H/L
            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)
            hl_ratio = np.log(high / low.replace(0.0, np.nan))
            # Simplified Corwin-Schultz: spread ≈ 2*(e^alpha - 1) / (1 + e^alpha)
            alpha = (hl_ratio.rolling(2, min_periods=2).mean()
                     - 0.5 * hl_ratio.rolling(2, min_periods=2).std())
            f["bid_ask_proxy"] = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

        # Order imbalance proxy (net buying pressure estimate)
        if "order_imbalance" in df.columns:
            f["order_imbalance"] = df["order_imbalance"].astype(float)
        else:
            # Use price position within H-L range as a proxy for order imbalance
            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)
            f["order_imbalance_proxy"] = (
                (close - low) / (high - low).replace(0.0, np.nan) - 0.5
            )

        # Trade intensity — only include when real microstructure data is available
        for col_name in ("trade_count", "large_trade_ratio", "tick_direction",
                         "price_impact", "effective_spread"):
            if col_name in df.columns:
                f[col_name] = df[col_name].astype(float)

        return f
