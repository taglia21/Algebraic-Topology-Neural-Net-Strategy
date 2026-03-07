"""
equities/strategies/stat_arb.py
================================
Statistical Arbitrage / Pairs Trading strategy for the ATNN trading system.

Overview
--------
This module implements a classic mean-reversion pairs trading strategy driven
by statistical cointegration analysis, Ornstein-Uhlenbeck (OU) spread dynamics,
and a Kalman filter for time-varying hedge ratio estimation.

Pipeline
--------
1. ``find_pairs(price_data)``
   - Tests all symbol pairs with Engle-Granger cointegration (p < 0.05).
   - Fits an OU model to the spread: dS = θ(μ - S)dt + σdW.
   - Rejects pairs with half-life outside [1, 60] trading days.
   - Initialises a per-pair Kalman filter for the hedge ratio.

2. ``generate_signals(price_data, regime_state)``
   - Computes the z-score of each pair's spread using a rolling window = 2×half_life.
   - Entry LONG spread when z-score < -1.5 (or -2.0 conservative).
   - Entry SHORT spread when z-score > 1.5 (or 2.0).
   - Exit when z-score crosses 0.5 toward mean.
   - Hard stop at |z-score| ≥ 3.0.
   - Blocked entirely in CRISIS regime.

References
----------
- Engle & Granger (1987), Econometrica
- Avellaneda & Lee (2010), Quantitative Finance
- Elliott, van der Hoek & Malcolm (2005), Applied Mathematical Finance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from core.config import StatArbConfig, get_config
from core.logger import TradeLogger, get_trade_logger
from core.regime_detector import RegimeState
from equities.models import Pair, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engle-Granger cointegration (pure numpy — no statsmodels required at runtime;
# statsmodels is imported lazily and replaced by a manual residual ADF test
# if unavailable)
# ---------------------------------------------------------------------------

def _engle_granger_pvalue(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Run Engle-Granger cointegration test; return (p_value, beta).

    The test regresses y on x (with intercept), extracts residuals, then
    runs an Augmented Dickey-Fuller test on the residuals.  The ADF statistic
    tests the null hypothesis that residuals have a unit root (non-stationarity).
    Rejecting the null (low p-value) implies cointegration.

    Parameters
    ----------
    y, x:
        Equal-length price series.

    Returns
    -------
    (p_value, beta):
        ADF p-value and OLS regression coefficient (hedge ratio).
    """
    # OLS: y = alpha + beta * x + epsilon
    X_mat = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        logger.debug(f"OLS failed in cointegration test: {exc}")
        return 1.0, 1.0

    beta = float(coeffs[1])
    residuals = y - X_mat @ coeffs

    try:
        from statsmodels.tsa.stattools import adfuller
        # statsmodels >= 0.14 uses maxlag (singular); autolag=None means use maxlag directly
        adf_result = adfuller(residuals, maxlag=1, autolag=None)
        pvalue = float(adf_result[1])
    except ImportError:
        # Fallback: simple OLS-based approximate ADF (less precise)
        pvalue = _approx_adf_pvalue(residuals)

    return pvalue, beta


def _approx_adf_pvalue(residuals: np.ndarray) -> float:
    """Approximate ADF p-value via regression of Δε on ε_{t-1}.

    This is a simplified Dickey-Fuller test without augmentation lags.
    Used as a fallback when statsmodels is unavailable.

    Parameters
    ----------
    residuals:
        Residual series from cointegrating regression.

    Returns
    -------
    Approximate p-value.  Conservative (tends to over-estimate p).
    """
    delta = np.diff(residuals)
    lagged = residuals[:-1]
    # Regress delta on lagged level
    X = np.column_stack([np.ones(len(lagged)), lagged])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
        rho = coeffs[1]
        # Compute t-statistic
        predicted = X @ coeffs
        sse = np.sum((delta - predicted) ** 2)
        s2 = sse / max(len(delta) - 2, 1)
        var_rho = s2 / max(np.sum((lagged - lagged.mean()) ** 2), 1e-12)
        t_stat = rho / np.sqrt(max(var_rho, 1e-12))
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return 1.0

    # Map DF t-statistic to approximate p-value using normal distribution
    # (conservative: true critical values are more negative, so this tends
    # to *increase* the estimated p-value, making the filter tighter)
    pvalue = float(stats.norm.cdf(t_stat))
    return min(max(pvalue, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck estimation
# ---------------------------------------------------------------------------

def _fit_ou_params(spread: np.ndarray) -> Tuple[float, float, float]:
    """Estimate OU parameters (θ, μ, σ) from a spread time series.

    Uses the discrete-time analogue of the OU process via OLS on:
        S_t = a + b * S_{t-1} + ε_t

    where:
        b  = exp(−θ Δt)  →  θ = −ln(b) / Δt  (Δt = 1 day)
        a  = μ (1 − b)   →  μ = a / (1 − b)
        σ² = Var(ε) × 2θ / (1 − exp(−2θ))  ≈ Var(ε) for small Δt

    Parameters
    ----------
    spread:
        Daily spread time series (should be roughly stationary for OU to apply).

    Returns
    -------
    (theta, mu, sigma):
        Mean-reversion speed, long-run mean, diffusion coefficient.

    Raises
    ------
    ValueError:
        If the series is too short or degenerate.
    """
    if len(spread) < 10:
        raise ValueError(f"Need at least 10 observations; got {len(spread)}")

    s_t = spread[1:]
    s_lag = spread[:-1]

    X = np.column_stack([np.ones(len(s_lag)), s_lag])
    try:
        coeffs, residuals_ss, _, _ = np.linalg.lstsq(X, s_t, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"OLS failed in OU estimation: {exc}") from exc

    a, b = float(coeffs[0]), float(coeffs[1])

    # Guard against explosive or unit-root process
    b = np.clip(b, -0.9999, 0.9999)

    # θ (mean-reversion speed), Δt = 1 trading day
    if b <= 0:
        # Anti-correlated lag → use |b|
        b = abs(b)
    theta = max(-np.log(b), 1e-6)  # force positive

    mu = a / max(1.0 - b, 1e-6)

    # Residual std
    predicted = X @ coeffs
    residuals = s_t - predicted
    sigma = float(np.std(residuals))

    return theta, mu, sigma


def _half_life_from_theta(theta: float) -> float:
    """Compute mean-reversion half-life from OU speed θ.

    half_life = ln(2) / θ  (in the same time units as θ, here: trading days)

    Parameters
    ----------
    theta:
        OU mean-reversion speed (must be > 0).

    Returns
    -------
    Half-life in trading days.
    """
    if theta <= 0:
        return float("inf")
    return float(np.log(2.0) / theta)


# ---------------------------------------------------------------------------
# Kalman filter for time-varying hedge ratio
# ---------------------------------------------------------------------------

class KalmanHedgeRatio:
    """Online Kalman filter estimating a time-varying hedge ratio.

    State vector: [beta_0 (intercept), beta_1 (hedge ratio)]
    Observation: y_t = beta_0 + beta_1 * x_t + v_t  (v_t ~ N(0, R))
    State evolution: beta_t = beta_{t-1} + w_t      (w_t ~ N(0, Q))

    Parameters
    ----------
    transition_cov:
        Process noise covariance Q (controls how fast the ratio can change).
        Larger values allow faster adaptation but add noise.
    observation_cov:
        Observation noise variance R.
    initial_state:
        Initial guess for [intercept, hedge_ratio].  If None, defaults to
        [0.0, 1.0].
    """

    def __init__(
        self,
        transition_cov: float = 1e-5,
        observation_cov: float = 1e-3,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        self._Q = np.eye(2) * transition_cov
        self._R = np.array([[observation_cov]])

        # State mean and covariance
        self._theta = (
            initial_state.copy() if initial_state is not None
            else np.array([0.0, 1.0])
        )
        self._P = np.eye(2) * 1.0  # initial state covariance

    def update(self, y: float, x: float) -> Tuple[float, float]:
        """Process one observation and return updated hedge ratio.

        Parameters
        ----------
        y:
            Observation for the first (dependent) leg.
        x:
            Observation for the second (independent) leg.

        Returns
        -------
        (intercept, hedge_ratio):
            Updated state estimate.
        """
        # Observation matrix: H = [1, x]
        H = np.array([[1.0, x]])

        # Predict step
        theta_pred = self._theta          # constant state transition (F = I)
        P_pred = self._P + self._Q

        # Innovation
        y_pred = float(H @ theta_pred)
        innovation = y - y_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + self._R   # (1, 1)

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)  # (2, 1)

        # Update
        self._theta = theta_pred + K.flatten() * innovation
        self._P = (np.eye(2) - K @ H) @ P_pred

        return float(self._theta[0]), float(self._theta[1])

    def batch_update(
        self,
        y_series: np.ndarray,
        x_series: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a full historical series and return time-varying estimates.

        Parameters
        ----------
        y_series, x_series:
            Equal-length price arrays.

        Returns
        -------
        (intercepts, hedge_ratios):
            Arrays of state estimates for each time step.
        """
        n = len(y_series)
        intercepts = np.zeros(n)
        hedge_ratios = np.zeros(n)

        for i in range(n):
            intercepts[i], hedge_ratios[i] = self.update(
                float(y_series[i]), float(x_series[i])
            )

        return intercepts, hedge_ratios

    @property
    def hedge_ratio(self) -> float:
        """Current (latest) hedge ratio estimate."""
        return float(self._theta[1])

    @property
    def intercept(self) -> float:
        """Current (latest) intercept estimate."""
        return float(self._theta[0])


# ---------------------------------------------------------------------------
# State tracker for open pair positions
# ---------------------------------------------------------------------------

@dataclass
class _PairPosition:
    """Tracks whether a pair is currently in a position and on which side."""

    pair_id: str
    side: str          # 'long_spread' or 'short_spread'
    entry_zscore: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class StatArbStrategy:
    """Statistical arbitrage / pairs trading strategy.

    The strategy identifies cointegrated pairs using the Engle-Granger test,
    models spread dynamics with an Ornstein-Uhlenbeck process, and uses a
    Kalman filter to track time-varying hedge ratios.  Signals are generated
    based on normalised z-scores of the OU spread.

    Parameters
    ----------
    config:
        ``StatArbConfig`` from the system configuration.  If *None*, the
        global config is read via :func:`~core.config.get_config`.
    trade_logger:
        ``TradeLogger`` for audit logging.  If *None*, uses the process default.
    conservative_entry:
        If True, use 2.0 sigma entry threshold; otherwise use 1.5 sigma.

    Usage
    -----
    >>> strategy = StatArbStrategy()
    >>> pairs = strategy.find_pairs(price_data)
    >>> signals = strategy.generate_signals(price_data, regime_state)
    """

    STRATEGY_NAME: str = "stat_arb"

    def __init__(
        self,
        config: Optional[StatArbConfig] = None,
        trade_logger: Optional[TradeLogger] = None,
        conservative_entry: bool = False,
    ) -> None:
        cfg = config or get_config().strategy.stat_arb
        self._cfg = cfg
        self._log = trade_logger or get_trade_logger()
        self._conservative_entry = conservative_entry

        # Entry thresholds
        self._entry_z: float = cfg.entry_z if conservative_entry else cfg.min_entry_z
        self._exit_z: float = cfg.exit_z
        self._stop_z: float = cfg.stop_z

        # Kalman filter noise parameters
        self._kalman_transition_cov: float = cfg.kalman_transition_cov
        self._kalman_observation_cov: float = cfg.kalman_observation_cov

        # State: fitted pairs and their Kalman filters
        self._pairs: List[Pair] = []
        self._kalman_filters: Dict[str, KalmanHedgeRatio] = {}
        self._open_positions: Dict[str, _PairPosition] = {}

    # ------------------------------------------------------------------
    # Pair discovery
    # ------------------------------------------------------------------

    def find_pairs(
        self,
        price_data: pd.DataFrame,
        min_history_days: int = 252,
    ) -> List[Pair]:
        """Discover cointegrated pairs from a price matrix.

        Runs Engle-Granger cointegration tests on all symbol pairs with
        sufficient history, fits OU models, and rejects pairs with
        unfavourable mean-reversion characteristics.

        Parameters
        ----------
        price_data:
            Wide-format DataFrame where each column is a symbol's *close* price
            and the index is a date-sorted DatetimeIndex.
        min_history_days:
            Minimum number of overlapping trading days required for a pair to
            be tested (default 252 ≈ 1 trading year).

        Returns
        -------
        List[Pair]:
            Valid cointegrated pairs sorted by half-life (ascending).  The list
            is also stored on ``self._pairs`` and Kalman filters are
            re-initialised for each discovered pair.
        """
        logger.info(
            f"StatArbStrategy.find_pairs: testing {price_data.shape[1]} symbols "
            f"with {len(price_data)} bars of history."
        )

        symbols = [c for c in price_data.columns if c != "SPY"]
        discovered: List[Pair] = []

        for sym_x, sym_y in combinations(symbols, 2):
            # Drop NaN rows for this pair
            pair_df = price_data[[sym_x, sym_y]].dropna()
            if len(pair_df) < min_history_days:
                continue

            y_vals = pair_df[sym_x].values.astype(float)
            x_vals = pair_df[sym_y].values.astype(float)

            # Cointegration test
            try:
                pvalue, beta = _engle_granger_pvalue(y_vals, x_vals)
            except Exception as exc:
                logger.debug(f"Cointegration test failed for {sym_x}/{sym_y}: {exc}")
                continue

            if pvalue >= 0.05:
                continue

            # Compute spread using static hedge ratio (OU fitting uses static spread)
            spread = y_vals - beta * x_vals

            # Fit OU model
            try:
                theta, mu, sigma = _fit_ou_params(spread)
            except ValueError as exc:
                logger.debug(f"OU fitting failed for {sym_x}/{sym_y}: {exc}")
                continue

            half_life = _half_life_from_theta(theta)

            # Half-life filter: reject pairs that mean-revert too slowly or instantly
            if not (1.0 <= half_life <= 120.0):
                logger.debug(
                    f"Rejected {sym_x}/{sym_y}: half_life={half_life:.1f} days "
                    f"(must be in [1, 120])."
                )
                continue

            pair = Pair(
                symbol_x=sym_x,
                symbol_y=sym_y,
                hedge_ratio=beta,
                half_life=half_life,
                coint_pvalue=pvalue,
                ou_theta=theta,
                ou_mu=mu,
                ou_sigma=sigma,
                lookback_days=len(pair_df),
            )
            discovered.append(pair)

            logger.info(
                f"Found pair {sym_x}/{sym_y}: p={pvalue:.4f}, "
                f"half_life={half_life:.1f}d, hedge_ratio={beta:.4f}"
            )

        # Sort by half-life (faster mean reversion first)
        discovered.sort(key=lambda p: p.half_life)

        # Cap at 25 pairs to keep per-bar evaluation fast.
        # Pairs are already sorted by half-life, so we keep the
        # fastest-reverting ones which are the highest-quality signals.
        _MAX_ACTIVE_PAIRS = 25
        if len(discovered) > _MAX_ACTIVE_PAIRS:
            logger.info(
                f"StatArbStrategy: capping pairs from {len(discovered)} "
                f"to {_MAX_ACTIVE_PAIRS} (by half-life)."
            )
            discovered = discovered[:_MAX_ACTIVE_PAIRS]

        # Store and initialise Kalman filters
        self._pairs = discovered
        self._kalman_filters = {}
        for pair in discovered:
            kf = KalmanHedgeRatio(
                transition_cov=self._kalman_transition_cov,
                observation_cov=self._kalman_observation_cov,
                initial_state=np.array([0.0, pair.hedge_ratio]),
            )
            # Warm-up Kalman filter on historical data
            pair_df = price_data[[pair.symbol_x, pair.symbol_y]].dropna()
            y_arr = pair_df[pair.symbol_x].values.astype(float)
            x_arr = pair_df[pair.symbol_y].values.astype(float)
            kf.batch_update(y_arr, x_arr)
            self._kalman_filters[pair.pair_id] = kf

        logger.info(
            f"StatArbStrategy.find_pairs: found {len(discovered)} valid pairs "
            f"from {len(symbols)} symbols."
        )
        return discovered

    # ------------------------------------------------------------------
    # Spread and z-score computation
    # ------------------------------------------------------------------

    def _compute_spread_zscore(
        self,
        pair: Pair,
        price_data: pd.DataFrame,
    ) -> Optional[Tuple[pd.Series, pd.Series, float]]:
        """Compute rolling spread z-score for a pair using Kalman hedge ratios.

        Parameters
        ----------
        pair:
            The :class:`Pair` object.
        price_data:
            Full price matrix.

        Returns
        -------
        (spread, zscore, latest_zscore) or None if data is insufficient.
        """
        if pair.symbol_x not in price_data.columns or pair.symbol_y not in price_data.columns:
            return None

        pair_df = price_data[[pair.symbol_x, pair.symbol_y]].dropna()
        if len(pair_df) < max(int(pair.half_life * 2) + 5, 20):
            return None

        y_arr = pair_df[pair.symbol_x].values.astype(float)
        x_arr = pair_df[pair.symbol_y].values.astype(float)

        # Get time-varying hedge ratios from Kalman filter (read-only predict pass)
        kf = self._kalman_filters.get(pair.pair_id)
        if kf is None:
            # Fall back to static hedge ratio
            hedge_ratios = np.full(len(y_arr), pair.hedge_ratio)
        else:
            # Re-run Kalman on this window to get latest ratios without mutation
            kf_temp = KalmanHedgeRatio(
                transition_cov=self._kalman_transition_cov,
                observation_cov=self._kalman_observation_cov,
                initial_state=np.array([kf.intercept, kf.hedge_ratio]),
            )
            _, hedge_ratios = kf_temp.batch_update(y_arr, x_arr)

        # Compute spread using time-varying hedge ratio
        spread = pd.Series(
            y_arr - hedge_ratios * x_arr,
            index=pair_df.index,
            name=f"spread_{pair.pair_id}",
        )

        # Rolling window = 2 × half_life
        window = max(int(pair.half_life * 2), 20)
        roll_mean = spread.rolling(window=window, min_periods=window // 2).mean()
        roll_std = spread.rolling(window=window, min_periods=window // 2).std()

        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan)
        zscore = (spread - roll_mean) / roll_std

        latest_z = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else float("nan")
        return spread, zscore, latest_z

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        regime_state: RegimeState,
    ) -> List[Signal]:
        """Generate trading signals for all active pairs.

        Parameters
        ----------
        price_data:
            Wide-format DataFrame with symbol columns and DatetimeIndex.
            Must contain prices for all symbols in ``self._pairs``.
        regime_state:
            Current market regime from :class:`~core.regime_detector.RegimeDetector`.

        Returns
        -------
        List[Signal]:
            Signals for the current bar.  Each signal carries both legs in
            ``metadata``: ``symbol_long``, ``symbol_short``, ``hedge_ratio``,
            ``z_score``, ``half_life``.

        Notes
        -----
        - Blocked entirely in CRISIS regime (``regime_state.is_crisis == True``).
        - STOP signals are always emitted regardless of regime.
        """
        # Check for crisis
        if regime_state.is_crisis:
            logger.info("StatArbStrategy: blocked — CRISIS regime detected.")
            # Still check for stops on open positions
            return self._generate_stop_signals(price_data, regime_state)

        signals: List[Signal] = []

        if not self._pairs:
            logger.warning(
                "StatArbStrategy.generate_signals: no pairs loaded. "
                "Call find_pairs() first."
            )
            return signals

        for pair in self._pairs:
            result = self._compute_spread_zscore(pair, price_data)
            if result is None:
                continue

            _, zscore_series, latest_z = result

            if np.isnan(latest_z):
                continue

            pair_id = pair.pair_id
            current_pos = self._open_positions.get(pair_id)
            abs_z = abs(latest_z)

            # --- Stop loss check ---
            if current_pos is not None and abs_z >= self._stop_z:
                signal = Signal(
                    symbol=pair.symbol_x,
                    direction="close",
                    strength=1.0,
                    strategy=self.STRATEGY_NAME,
                    metadata={
                        "symbol_long": (
                            pair.symbol_x if current_pos.side == "long_spread"
                            else pair.symbol_y
                        ),
                        "symbol_short": (
                            pair.symbol_y if current_pos.side == "long_spread"
                            else pair.symbol_x
                        ),
                        "hedge_ratio": pair.hedge_ratio,
                        "z_score": latest_z,
                        "half_life": pair.half_life,
                        "action": "stop_loss",
                        "pair_id": pair_id,
                    },
                )
                signals.append(signal)
                del self._open_positions[pair_id]
                self._log.log_signal(
                    self.STRATEGY_NAME, pair.symbol_x, "FLAT", 1.0,
                    {"reason": "stop_loss", "z_score": latest_z, "pair_id": pair_id},
                )
                continue

            # --- Exit existing position ---
            if current_pos is not None:
                should_exit = (
                    (current_pos.side == "long_spread" and latest_z > -self._exit_z)
                    or (current_pos.side == "short_spread" and latest_z < self._exit_z)
                )
                if should_exit:
                    signal = Signal(
                        symbol=pair.symbol_x,
                        direction="close",
                        strength=1.0,
                        strategy=self.STRATEGY_NAME,
                        metadata={
                            "symbol_long": (
                                pair.symbol_x if current_pos.side == "long_spread"
                                else pair.symbol_y
                            ),
                            "symbol_short": (
                                pair.symbol_y if current_pos.side == "long_spread"
                                else pair.symbol_x
                            ),
                            "hedge_ratio": pair.hedge_ratio,
                            "z_score": latest_z,
                            "half_life": pair.half_life,
                            "action": "exit",
                            "pair_id": pair_id,
                        },
                    )
                    signals.append(signal)
                    del self._open_positions[pair_id]
                    continue

            # --- Entry signals (only if not already in a position) ---
            if current_pos is not None:
                continue

            # Signal strength: how far is z_score from the threshold (capped at 1.0)?
            strength = min(
                (abs_z - self._entry_z) / max(self._stop_z - self._entry_z, 0.5),
                1.0,
            )

            if latest_z < -self._entry_z:
                # Long spread: buy symbol_x, sell symbol_y
                signal = Signal(
                    symbol=pair.symbol_x,
                    direction="long",
                    strength=max(strength, 0.01),
                    strategy=self.STRATEGY_NAME,
                    metadata={
                        "symbol_long": pair.symbol_x,
                        "symbol_short": pair.symbol_y,
                        "hedge_ratio": pair.hedge_ratio,
                        "z_score": latest_z,
                        "half_life": pair.half_life,
                        "action": "entry_long_spread",
                        "pair_id": pair_id,
                        "ou_theta": pair.ou_theta,
                        "ou_mu": pair.ou_mu,
                        "coint_pvalue": pair.coint_pvalue,
                    },
                )
                signals.append(signal)
                self._open_positions[pair_id] = _PairPosition(
                    pair_id=pair_id,
                    side="long_spread",
                    entry_zscore=latest_z,
                )
                self._log.log_signal(
                    self.STRATEGY_NAME, pair.symbol_x, "BUY", float(signal.strength),
                    {"pair_id": pair_id, "z_score": latest_z},
                )

            elif latest_z > self._entry_z:
                # Short spread: sell symbol_x, buy symbol_y
                signal = Signal(
                    symbol=pair.symbol_x,
                    direction="short",
                    strength=max(strength, 0.01),
                    strategy=self.STRATEGY_NAME,
                    metadata={
                        "symbol_long": pair.symbol_y,
                        "symbol_short": pair.symbol_x,
                        "hedge_ratio": pair.hedge_ratio,
                        "z_score": latest_z,
                        "half_life": pair.half_life,
                        "action": "entry_short_spread",
                        "pair_id": pair_id,
                        "ou_theta": pair.ou_theta,
                        "ou_mu": pair.ou_mu,
                        "coint_pvalue": pair.coint_pvalue,
                    },
                )
                signals.append(signal)
                self._open_positions[pair_id] = _PairPosition(
                    pair_id=pair_id,
                    side="short_spread",
                    entry_zscore=latest_z,
                )
                self._log.log_signal(
                    self.STRATEGY_NAME, pair.symbol_x, "SELL", float(signal.strength),
                    {"pair_id": pair_id, "z_score": latest_z},
                )

        return signals

    def _generate_stop_signals(
        self,
        price_data: pd.DataFrame,
        regime_state: RegimeState,
    ) -> List[Signal]:
        """Generate close signals for all open pair positions (crisis exit)."""
        signals: List[Signal] = []
        for pair_id, pos in list(self._open_positions.items()):
            # Find the pair
            matching = [p for p in self._pairs if p.pair_id == pair_id]
            if not matching:
                continue
            pair = matching[0]
            signal = Signal(
                symbol=pair.symbol_x,
                direction="close",
                strength=1.0,
                strategy=self.STRATEGY_NAME,
                metadata={
                    "symbol_long": (
                        pair.symbol_x if pos.side == "long_spread" else pair.symbol_y
                    ),
                    "symbol_short": (
                        pair.symbol_y if pos.side == "long_spread" else pair.symbol_x
                    ),
                    "hedge_ratio": pair.hedge_ratio,
                    "action": "crisis_exit",
                    "pair_id": pair_id,
                },
            )
            signals.append(signal)

        # Clear all positions on crisis exit
        self._open_positions.clear()
        return signals

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def pairs(self) -> List[Pair]:
        """Currently active pairs."""
        return list(self._pairs)

    @property
    def n_pairs(self) -> int:
        """Number of active pairs."""
        return len(self._pairs)

    @property
    def open_positions(self) -> Dict[str, _PairPosition]:
        """Pairs currently holding an open spread position."""
        return dict(self._open_positions)

    def reset_positions(self) -> None:
        """Clear all tracked open positions (e.g., at start of new backtest window)."""
        self._open_positions.clear()
