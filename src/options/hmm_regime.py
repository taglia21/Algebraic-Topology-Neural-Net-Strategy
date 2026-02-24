"""
HMM Regime Classifier (Phase G, Item 5)
=========================================

Gaussian HMM with 3 hidden states trained on a rolling 252-bar window
of [log_return, realized_vol, volume_z]:

    State 0 — Low-vol trending   (trade long/short directionally)
    State 1 — High-vol choppy    (no new positions)
    State 2 — Crisis             (sell premium / tail protection)

Gate all trades: only open new positions in state 0 (trending) or
state 2 (crisis = sell premium).

Uses ``hmmlearn.GaussianHMM`` when available; falls back to a simple
volatility-threshold classifier when hmmlearn is not installed.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["HMMRegimeClassifier", "RegimeLabel"]

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    GaussianHMM = None
    HMM_AVAILABLE = False


class RegimeLabel:
    """Constants for regime states."""
    TRENDING = 0
    CHOPPY = 1
    CRISIS = 2

    NAMES = {0: "low_vol_trending", 1: "high_vol_choppy", 2: "crisis"}
    TRADEABLE = {0, 2}  # only these allow new positions


class HMMRegimeClassifier:
    """3-state Gaussian HMM regime classifier.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3).
    lookback : int
        Rolling training window in bars (default 252).
    retrain_every : int
        Re-fit the HMM every N new bars (default 20).
    vol_window : int
        Window for realized vol calculation (default 20).
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 252,
        retrain_every: int = 20,
        vol_window: int = 20,
    ):
        self.n_states = n_states
        self.lookback = lookback
        self.retrain_every = retrain_every
        self.vol_window = vol_window

        self._model: Optional[object] = None
        self._fitted: bool = False
        self._bars_since_fit: int = 0
        self._current_state: int = RegimeLabel.TRENDING
        self._confidence: float = 0.0

        # History buffers
        self._log_returns: List[float] = []
        self._volumes: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        log_return: float,
        volume: float,
    ) -> int:
        """Add a new bar and return the current regime label.

        Parameters
        ----------
        log_return : float
            Log return of the bar.
        volume : float
            Volume of the bar (raw).

        Returns
        -------
        int
            Regime label (0, 1, or 2).
        """
        self._log_returns.append(log_return)
        self._volumes.append(volume)
        self._bars_since_fit += 1

        # Keep only lookback + buffer
        max_keep = self.lookback + 50
        if len(self._log_returns) > max_keep:
            self._log_returns = self._log_returns[-max_keep:]
            self._volumes = self._volumes[-max_keep:]

        n = len(self._log_returns)
        if n < max(self.vol_window + 1, 30):
            return self._current_state

        # Retrain periodically
        if not self._fitted or self._bars_since_fit >= self.retrain_every:
            self._fit()

        # Predict current state
        self._current_state, self._confidence = self._predict_current()
        return self._current_state

    @property
    def current_state(self) -> int:
        """Current regime label."""
        return self._current_state

    @property
    def state_name(self) -> str:
        """Human-readable name of current state."""
        return RegimeLabel.NAMES.get(self._current_state, "unknown")

    @property
    def confidence(self) -> float:
        """Confidence in the current state classification (0-1)."""
        return self._confidence

    @property
    def is_tradeable(self) -> bool:
        """Whether the current regime allows new positions."""
        return self._current_state in RegimeLabel.TRADEABLE

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_features(self) -> np.ndarray:
        """Build [log_return, realized_vol, volume_z] feature matrix."""
        rets = np.array(self._log_returns)
        vols_raw = np.array(self._volumes)
        n = len(rets)

        # Realized vol (rolling std of log returns)
        rv = np.full(n, np.nan)
        for i in range(self.vol_window, n):
            rv[i] = np.std(rets[i - self.vol_window : i])

        # Volume z-score
        vol_z = np.full(n, 0.0)
        for i in range(self.vol_window, n):
            window = vols_raw[i - self.vol_window : i]
            mu, sig = np.mean(window), np.std(window)
            vol_z[i] = (vols_raw[i] - mu) / sig if sig > 0 else 0.0

        # Drop NaN rows (first vol_window)
        valid = ~np.isnan(rv)
        X = np.column_stack([rets[valid], rv[valid], vol_z[valid]])
        return X

    # ------------------------------------------------------------------
    # Fit & predict
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Train the HMM on the rolling window."""
        X = self._build_features()
        if len(X) < 30:
            return

        # Use only last `lookback` rows
        X = X[-self.lookback:]

        if HMM_AVAILABLE:
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=50,
                    random_state=42,
                    verbose=False,
                )
                model.fit(X)
                self._model = model
                self._fitted = True
                self._bars_since_fit = 0
                logger.info("HMM fitted on %d bars", len(X))
            except Exception as exc:
                logger.warning("HMM fit failed: %s — using fallback", exc)
                self._fitted = True  # use fallback
                self._model = None
        else:
            self._fitted = True
            self._model = None
            logger.debug("hmmlearn not installed — using threshold fallback")

        self._bars_since_fit = 0

    def _predict_current(self) -> Tuple[int, float]:
        """Predict current regime state."""
        X = self._build_features()
        if len(X) < 5:
            return RegimeLabel.TRENDING, 0.5

        if self._model is not None and HMM_AVAILABLE:
            try:
                proba = self._model.predict_proba(X[-1:])
                state = int(np.argmax(proba[0]))
                conf = float(proba[0][state])

                # Map HMM states to semantic labels by volatility
                # State with lowest mean volatility → TRENDING
                # State with highest → CRISIS, middle → CHOPPY
                means = self._model.means_
                vol_col = 1  # realized_vol column
                vol_means = means[:, vol_col]
                order = np.argsort(vol_means)
                state_map = {int(order[0]): RegimeLabel.TRENDING,
                             int(order[1]): RegimeLabel.CHOPPY,
                             int(order[2]): RegimeLabel.CRISIS}
                mapped_state = state_map.get(state, RegimeLabel.TRENDING)
                return mapped_state, conf
            except Exception as exc:
                logger.debug("HMM predict failed: %s", exc)

        # Fallback: simple threshold
        return self._threshold_classify(X)

    def _threshold_classify(self, X: np.ndarray) -> Tuple[int, float]:
        """Simple volatility-threshold fallback."""
        recent_vol = float(np.mean(X[-5:, 1]))  # avg recent realized vol
        all_vol = float(np.mean(X[:, 1]))

        if recent_vol > 2.0 * all_vol:
            return RegimeLabel.CRISIS, 0.7
        elif recent_vol > 1.3 * all_vol:
            return RegimeLabel.CHOPPY, 0.6
        else:
            return RegimeLabel.TRENDING, 0.65
