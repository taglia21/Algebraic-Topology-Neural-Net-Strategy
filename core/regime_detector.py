"""
core/regime_detector.py
========================
HMM-based market-regime classifier for the ATNN trading system.

The detector trains a 3-state Gaussian HMM on SPY price/volume data and maps
each latent state to one of three semantic regimes:

    BULL     — low volatility, positive drift
    SIDEWAYS — low volatility, flat/no drift
    BEAR     — high volatility, negative drift

Secondary overlay signals augment the HMM output:

    VIX level     : LOW (<15) | NORMAL (15–25) | ELEVATED (25–35) | CRISIS (>35)
    ADX strength  : measures directional trend intensity

Usage
-----
    from core.regime_detector import RegimeDetector, RegimeState

    detector = RegimeDetector()
    detector.fit(spy_daily_ohlcv)          # pd.DataFrame with OHLCV columns
    state: RegimeState = detector.predict(spy_daily_ohlcv)

    print(state.regime)       # "BULL"
    print(state.confidence)   # 0.87
    print(state.is_crisis)    # False

Requirements
------------
    hmmlearn >= 0.3
    numpy, pandas (assumed available)
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TRAINING_DAYS: int = 60      # Minimum bars required for a reliable fit
N_HMM_STATES: int = 3
HMM_N_ITER: int = 200
HMM_RANDOM_STATE: int = 42

# VIX thresholds
VIX_LOW_THRESHOLD: float = 15.0
VIX_NORMAL_THRESHOLD: float = 25.0
VIX_ELEVATED_THRESHOLD: float = 35.0

# ADX thresholds
ADX_TRENDING_THRESHOLD: float = 25.0    # > 25 = directional trend
ADX_STRONG_THRESHOLD: float = 40.0     # > 40 = strong trend


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    """Three-state market regime labels."""
    BULL = "BULL"
    SIDEWAYS = "SIDEWAYS"
    BEAR = "BEAR"
    UNKNOWN = "UNKNOWN"  # Returned when insufficient history


class VIXLevel(str, Enum):
    """VIX-based fear / complacency classification."""
    LOW = "LOW"           # VIX < 15
    NORMAL = "NORMAL"     # 15 <= VIX < 25
    ELEVATED = "ELEVATED" # 25 <= VIX < 35
    CRISIS = "CRISIS"     # VIX >= 35
    UNKNOWN = "UNKNOWN"   # VIX data not available


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Complete regime snapshot returned by :meth:`RegimeDetector.predict`.

    Attributes
    ----------
    regime:
        The dominant market regime (BULL / SIDEWAYS / BEAR).
    confidence:
        HMM posterior probability for the predicted state, in (0, 1].
        Values above 0.70 are considered high-confidence.
    vix_level:
        VIX-based fear classification.
    vix_value:
        Raw VIX observation used for classification, or NaN if unavailable.
    adx:
        Average Directional Index (0–100).  > 25 indicates a trending market.
    is_trending:
        True when ADX > 25 (strong directional move).
    is_crisis:
        True when VIX >= 35 OR regime is BEAR with confidence > 0.80.
    regime_probs:
        Mapping of Regime → posterior probability for the latest bar.
    n_training_bars:
        Number of bars used to train the model.
    """
    regime: Regime
    confidence: float
    vix_level: VIXLevel
    vix_value: float
    adx: float
    is_trending: bool
    is_crisis: bool
    regime_probs: Dict[str, float]
    n_training_bars: int


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class _FeatureBuilder:
    """Build the feature matrix used to train and query the HMM.

    Features (all computed on *daily* bars):

        1. ``returns``       — 1-day log return
        2. ``realized_vol``  — 20-day annualised realised volatility
        3. ``volume_ratio``  — today's volume / 20-day average volume

    Parameters
    ----------
    vol_window:
        Rolling window for realised volatility (default 20 days).
    vol_norm_window:
        Rolling window for volume normalisation (default 20 days).
    """

    def __init__(self, vol_window: int = 20, vol_norm_window: int = 20) -> None:
        self.vol_window = vol_window
        self.vol_norm_window = vol_norm_window

    def build(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Compute features from an OHLCV DataFrame.

        Parameters
        ----------
        price_data:
            DataFrame with at minimum a ``close`` column.  Optional columns:
            ``volume``, ``vix``.  Column names are case-insensitive.

        Returns
        -------
        pd.DataFrame
            Original data plus feature columns; NaN rows are **not** dropped
            here (the caller decides how to handle them).

        Raises
        ------
        KeyError
            If the required ``close`` column is absent.
        """
        df = price_data.copy()

        # Normalise column names to lower-case
        df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            raise KeyError(
                "price_data must contain a 'close' column; "
                f"found columns: {list(df.columns)}"
            )

        close: pd.Series = df["close"].astype(float)

        # Feature 1: 1-day log returns
        df["returns"] = np.log(close / close.shift(1))

        # Feature 2: 20-day realised volatility (annualised)
        df["realized_vol"] = (
            df["returns"]
            .rolling(self.vol_window, min_periods=self.vol_window)
            .std()
            * np.sqrt(252)
        )

        # Feature 3: volume ratio
        if "volume" in df.columns:
            volume: pd.Series = df["volume"].astype(float).replace(0, np.nan)
            vol_ma = volume.rolling(self.vol_norm_window, min_periods=1).mean()
            df["volume_ratio"] = (volume / vol_ma).clip(upper=5.0)  # cap outliers
        else:
            df["volume_ratio"] = 1.0  # neutral when volume unavailable

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """Extract the 3-column feature array from a prepared DataFrame.

        Parameters
        ----------
        df:
            Output of :meth:`build`.

        Returns
        -------
        X:
            (n, 3) float array of valid (non-NaN) observations.
        valid_index:
            The DataFrame index for the rows included in X.
        """
        cols = ["returns", "realized_vol", "volume_ratio"]
        valid_mask = df[cols].notna().all(axis=1)
        valid_df = df.loc[valid_mask, cols]
        return valid_df.values.astype(float), valid_df.index


# ---------------------------------------------------------------------------
# ADX calculation (pure Python / NumPy — no TA-Lib dependency)
# ---------------------------------------------------------------------------

def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute the Average Directional Index.

    Parameters
    ----------
    high, low, close:
        Price series aligned by index.
    period:
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ADX values (0–100).  The first ``2*period`` bars will be NaN.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    # True range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smoothing
    def _wilder_smooth(series: pd.Series, n: int) -> pd.Series:
        smoothed = series.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
        return smoothed

    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, period) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_smooth(dx, period)

    return adx


def _classify_vix(vix_value: float) -> VIXLevel:
    """Return the VIX classification bucket for a scalar VIX observation."""
    if np.isnan(vix_value):
        return VIXLevel.UNKNOWN
    if vix_value < VIX_LOW_THRESHOLD:
        return VIXLevel.LOW
    if vix_value < VIX_NORMAL_THRESHOLD:
        return VIXLevel.NORMAL
    if vix_value < VIX_ELEVATED_THRESHOLD:
        return VIXLevel.ELEVATED
    return VIXLevel.CRISIS


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeDetector:
    """HMM-based market regime classifier.

    The detector is trained on SPY (or any broad market index) and applied
    market-wide.  It uses a 3-state Gaussian HMM over [returns, realised_vol,
    volume_ratio] to identify BULL / SIDEWAYS / BEAR environments.

    Parameters
    ----------
    n_states:
        Number of HMM latent states (default 3).
    n_iter:
        Maximum EM iterations for HMM fitting.
    random_state:
        Seed for reproducibility.
    vol_window:
        Rolling window used for realised-volatility computation.
    adx_period:
        Period for ADX calculation (default 14).
    """

    def __init__(
        self,
        n_states: int = N_HMM_STATES,
        n_iter: int = HMM_N_ITER,
        random_state: int = HMM_RANDOM_STATE,
        vol_window: int = 20,
        adx_period: int = 14,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.vol_window = vol_window
        self.adx_period = adx_period

        self._model: Optional[object] = None  # hmmlearn.GaussianHMM
        self._feature_builder = _FeatureBuilder(vol_window=vol_window)
        self._state_to_regime: Dict[int, Regime] = {}
        self._is_fitted: bool = False
        self._n_training_bars: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, price_data: pd.DataFrame) -> "RegimeDetector":
        """Train the HMM on historical price data.

        The DataFrame must contain at minimum a ``close`` column, and
        optionally ``high``, ``low``, ``volume`` columns.  All columns are
        case-insensitive.

        Parameters
        ----------
        price_data:
            Daily OHLCV data for SPY (or a broad-market proxy).

        Returns
        -------
        self
            Returns the fitted detector for chaining.

        Raises
        ------
        ValueError
            If fewer than ``MIN_TRAINING_DAYS`` valid observations are
            available after feature construction.
        ImportError
            If hmmlearn is not installed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError(
                "hmmlearn is required for RegimeDetector. "
                "Install it with: pip install hmmlearn"
            ) from exc

        logger.info("RegimeDetector.fit: building features ...")
        df = self._feature_builder.build(price_data)
        X, valid_index = self._feature_builder.get_feature_matrix(df)

        if len(X) < MIN_TRAINING_DAYS:
            raise ValueError(
                f"RegimeDetector requires at least {MIN_TRAINING_DAYS} valid "
                f"observations for reliable fitting; got {len(X)}. "
                "Provide more historical data."
            )

        logger.info(
            "RegimeDetector.fit: training GaussianHMM "
            f"(n_states={self.n_states}, n_samples={len(X)}) ..."
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress convergence warnings during fit
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(X)

        self._model = model
        self._n_training_bars = len(X)
        self._state_to_regime = self._map_states(model.means_)

        self._is_fitted = True
        logger.info(
            "RegimeDetector.fit: done.  State → Regime mapping: "
            + str({k: v.value for k, v in self._state_to_regime.items()})
        )

        return self

    # ------------------------------------------------------------------
    # State mapping
    # ------------------------------------------------------------------

    def _map_states(self, means: np.ndarray) -> Dict[int, Regime]:
        """Map the three latent HMM states to semantic regime labels.

        Mapping logic:
            - The state with the most negative mean return and highest
              volatility is BEAR.
            - The state with the most positive mean return is BULL.
            - The remaining state is SIDEWAYS.

        Parameters
        ----------
        means:
            (n_states, n_features) array of learned state means.
            Feature order: [returns, realized_vol, volume_ratio].

        Returns
        -------
        dict mapping HMM state int → Regime enum.
        """
        returns_idx = 0
        vol_idx = 1

        state_indices = list(range(self.n_states))

        # BEAR: worst combination of negative return and high volatility.
        # Score: penalise by (−return) + volatility
        bear_scores = [
            -means[i, returns_idx] + means[i, vol_idx]
            for i in state_indices
        ]
        bear_state = int(np.argmax(bear_scores))

        remaining = [i for i in state_indices if i != bear_state]

        # BULL: highest return among remaining states
        bull_state = max(remaining, key=lambda i: means[i, returns_idx])

        # SIDEWAYS: what's left
        sideways_candidates = [i for i in remaining if i != bull_state]
        if not sideways_candidates:
            # Edge case: n_states == 2 (shouldn't happen with default params)
            sideways_state = bear_state  # fallback
        else:
            sideways_state = sideways_candidates[0]

        mapping = {
            bear_state: Regime.BEAR,
            bull_state: Regime.BULL,
            sideways_state: Regime.SIDEWAYS,
        }

        logger.debug(
            "State means (returns, vol): "
            + ", ".join(
                f"state_{i}=({means[i, returns_idx]:.4f}, {means[i, vol_idx]:.4f})"
                for i in range(self.n_states)
            )
        )

        return mapping

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, price_data: pd.DataFrame) -> RegimeState:
        """Predict the current market regime from the latest available data.

        The method requires the HMM to be fitted first (:meth:`fit`).  If
        fewer than ``MIN_TRAINING_DAYS`` valid bars are available, it returns
        a :class:`RegimeState` with ``regime=UNKNOWN``.

        Parameters
        ----------
        price_data:
            Recent daily OHLCV data.  The most recent bar is used for the
            final prediction, but the full series is needed to compute
            features.  Should contain at minimum ``close`` and optionally
            ``high``, ``low``, ``volume``, ``vix`` columns.

        Returns
        -------
        RegimeState
            Current regime snapshot.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "RegimeDetector has not been fitted yet. Call fit() first."
            )

        df = self._feature_builder.build(price_data)
        X, valid_index = self._feature_builder.get_feature_matrix(df)

        if len(X) < MIN_TRAINING_DAYS:
            logger.warning(
                f"RegimeDetector.predict: only {len(X)} valid bars available "
                f"(need {MIN_TRAINING_DAYS}). Returning UNKNOWN regime."
            )
            return RegimeState(
                regime=Regime.UNKNOWN,
                confidence=0.0,
                vix_level=VIXLevel.UNKNOWN,
                vix_value=float("nan"),
                adx=float("nan"),
                is_trending=False,
                is_crisis=False,
                regime_probs={r.value: 0.0 for r in Regime},
                n_training_bars=self._n_training_bars,
            )

        # HMM posterior probabilities
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posteriors: np.ndarray = self._model.predict_proba(X)  # (n, n_states)

        latest_posteriors = posteriors[-1]  # shape (n_states,)

        # Map to regime probs
        regime_probs: Dict[str, float] = {r.value: 0.0 for r in Regime}
        for state_idx in range(self.n_states):
            r = self._state_to_regime.get(state_idx, Regime.UNKNOWN)
            regime_probs[r.value] = float(latest_posteriors[state_idx])

        # Determine the dominant regime
        best_state = int(np.argmax(latest_posteriors))
        regime = self._state_to_regime.get(best_state, Regime.UNKNOWN)
        confidence = float(latest_posteriors[best_state])

        # --- Secondary signals ---
        # VIX
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]
        if "vix" in df_lower.columns:
            vix_series = df_lower["vix"].dropna()
            vix_value = float(vix_series.iloc[-1]) if len(vix_series) > 0 else float("nan")
        else:
            vix_value = float("nan")

        vix_level = _classify_vix(vix_value)

        # ADX
        adx_value = float("nan")
        if all(c in df_lower.columns for c in ["high", "low", "close"]):
            try:
                adx_series = _compute_adx(
                    df_lower["high"],
                    df_lower["low"],
                    df_lower["close"],
                    period=self.adx_period,
                )
                adx_clean = adx_series.dropna()
                if len(adx_clean) > 0:
                    adx_value = float(adx_clean.iloc[-1])
            except Exception as exc:
                logger.warning(f"ADX computation failed: {exc}")

        is_trending = (
            not np.isnan(adx_value) and adx_value > ADX_TRENDING_THRESHOLD
        )

        # Crisis flag: only on extreme VIX.  Normal bear regimes should NOT
        # trigger crisis mode — the separate BEAR allocations handle those.
        is_crisis = vix_level == VIXLevel.CRISIS

        state = RegimeState(
            regime=regime,
            confidence=round(confidence, 6),
            vix_level=vix_level,
            vix_value=round(vix_value, 2) if not np.isnan(vix_value) else float("nan"),
            adx=round(adx_value, 2) if not np.isnan(adx_value) else float("nan"),
            is_trending=is_trending,
            is_crisis=is_crisis,
            regime_probs={k: round(v, 6) for k, v in regime_probs.items()},
            n_training_bars=self._n_training_bars,
        )

        return state

    def predict_series(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for every bar in *price_data*.

        Useful for back-testing: returns a DataFrame indexed like *price_data*
        with columns ``regime``, ``confidence``, ``hmm_state``, and one
        probability column per regime.

        Parameters
        ----------
        price_data:
            Daily OHLCV data.

        Returns
        -------
        pd.DataFrame
            Regime labels and posteriors for each valid bar.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "RegimeDetector has not been fitted yet. Call fit() first."
            )

        df = self._feature_builder.build(price_data)
        X, valid_index = self._feature_builder.get_feature_matrix(df)

        if len(X) == 0:
            return pd.DataFrame()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states: np.ndarray = self._model.predict(X)
            posteriors: np.ndarray = self._model.predict_proba(X)

        result = pd.DataFrame(index=valid_index)
        result["hmm_state"] = states
        result["regime"] = [self._state_to_regime.get(s, Regime.UNKNOWN).value for s in states]
        result["confidence"] = posteriors.max(axis=1)

        for i in range(self.n_states):
            r = self._state_to_regime.get(i, Regime.UNKNOWN)
            result[f"prob_{r.value}"] = posteriors[:, i]

        return result

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_regime_state(self, price_data: pd.DataFrame) -> RegimeState:
        """Alias for :meth:`predict`; provided for semantic clarity."""
        return self.predict(price_data)

    @property
    def is_fitted(self) -> bool:
        """True if the HMM has been successfully trained."""
        return self._is_fitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Persist the fitted model to disk.

        Parameters
        ----------
        filepath:
            Destination path.  Parent directories are created if absent.

        Raises
        ------
        RuntimeError
            If called before the model is fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted RegimeDetector.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "state_to_regime": self._state_to_regime,
            "n_training_bars": self._n_training_bars,
            "n_states": self.n_states,
            "vol_window": self.vol_window,
            "adx_period": self.adx_period,
        }
        with open(filepath, "wb") as fh:
            pickle.dump(payload, fh)

        logger.info(f"RegimeDetector saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "RegimeDetector":
        """Load a previously saved detector from disk.

        Parameters
        ----------
        filepath:
            Path to a ``pickle`` file created by :meth:`save`.

        Returns
        -------
        RegimeDetector
            Fully fitted detector ready for :meth:`predict`.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"No saved RegimeDetector found at {filepath!r}")

        with open(filepath, "rb") as fh:
            payload = pickle.load(fh)

        detector = cls(
            n_states=payload["n_states"],
            vol_window=payload["vol_window"],
            adx_period=payload["adx_period"],
        )
        detector._model = payload["model"]
        detector._state_to_regime = payload["state_to_regime"]
        detector._n_training_bars = payload["n_training_bars"]
        detector._is_fitted = True

        logger.info(f"RegimeDetector loaded from {filepath}")
        return detector
