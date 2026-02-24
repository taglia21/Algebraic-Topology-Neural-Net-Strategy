"""
Kalman Price Filter (Phase G, Item 4)
======================================

1-D Kalman filter tracking dynamic mid-price.

    State:  x_k = price (scalar)
    Model:  x_k = x_{k-1} + w,   w ~ N(0, Q)
    Obs:    z_k = x_k + v,        v ~ N(0, R)

Exposes ``filtered_price`` (posterior mean) and ``innovation``
(z_k - x̂_k|k-1, the prediction residual).

Usage
-----
    kf = KalmanPriceFilter(Q=0.01, R=1.0)
    for price in price_stream:
        kf.update(price)
        print(kf.filtered_price, kf.innovation)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["KalmanPriceFilter"]


class KalmanPriceFilter:
    """1-D Kalman filter for real-time mid-price tracking.

    Parameters
    ----------
    Q : float
        Process noise variance (default 0.01).
    R : float
        Observation noise variance (default 1.0).
    initial_estimate : float or None
        Initial state estimate; set on first ``update()`` if None.
    initial_variance : float
        Initial error variance (default 1.0).
    """

    def __init__(
        self,
        Q: float = 0.01,
        R: float = 1.0,
        initial_estimate: Optional[float] = None,
        initial_variance: float = 1.0,
    ):
        self.Q = Q
        self.R = R

        # State
        self._x: Optional[float] = initial_estimate  # posterior mean
        self._P: float = initial_variance            # posterior variance
        self._innovation: float = 0.0                # last innovation
        self._K: float = 0.0                         # last Kalman gain
        self._initialized: bool = initial_estimate is not None
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, observation: float) -> float:
        """Incorporate a new price observation.

        Parameters
        ----------
        observation : float
            Raw observed mid-price.

        Returns
        -------
        float
            Filtered (posterior) price estimate.
        """
        if not self._initialized:
            self._x = observation
            self._P = self.R
            self._initialized = True
            self._innovation = 0.0
            self._n_updates = 1
            return self._x

        # Predict step
        x_pred = self._x          # state transition is identity
        P_pred = self._P + self.Q

        # Innovation
        self._innovation = observation - x_pred

        # Kalman gain
        S = P_pred + self.R
        self._K = P_pred / S if S > 0 else 0.0

        # Update step
        self._x = x_pred + self._K * self._innovation
        self._P = (1.0 - self._K) * P_pred

        self._n_updates += 1
        return self._x

    @property
    def filtered_price(self) -> float:
        """Current filtered (posterior) price estimate."""
        return self._x if self._x is not None else 0.0

    @property
    def innovation(self) -> float:
        """Last innovation (residual): z_k - x̂_k|k-1."""
        return self._innovation

    @property
    def kalman_gain(self) -> float:
        """Current Kalman gain."""
        return self._K

    @property
    def variance(self) -> float:
        """Current posterior variance."""
        return self._P

    @property
    def n_updates(self) -> int:
        """Total number of updates processed."""
        return self._n_updates

    def reset(self, estimate: Optional[float] = None) -> None:
        """Reset the filter state."""
        self._x = estimate
        self._P = self.R
        self._innovation = 0.0
        self._K = 0.0
        self._initialized = estimate is not None
        self._n_updates = 0
