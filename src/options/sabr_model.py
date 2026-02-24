"""
Phase P — SABR Volatility Model.

Item 10: SABRVolatilityModel — Hagan SABR closed-form, beta=0.5, Nelder-Mead calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class SABRParams:
    """SABR model parameters."""
    alpha: float = 0.2    # initial vol-of-vol
    beta: float = 0.5     # CEV exponent (fixed)
    rho: float = -0.3     # correlation between asset and vol
    nu: float = 0.4       # vol-of-vol


@dataclass
class SABRCalibrationResult:
    """Result of SABR calibration."""
    params: SABRParams = field(default_factory=SABRParams)
    rmse: float = 0.0
    n_strikes: int = 0
    success: bool = False


class SABRVolatilityModel:
    """SABR stochastic volatility model (Hagan et al. 2002).

    dF = alpha * F^beta * dW1
    dalpha = nu * alpha * dW2
    <dW1, dW2> = rho * dt

    Uses Hagan's closed-form implied volatility approximation.
    Calibration via Nelder-Mead optimization.
    """

    def __init__(self, beta: float = 0.5):
        """
        Args:
            beta: CEV exponent (typically fixed at 0.5 for equities).
        """
        self.beta = beta
        self._params: Optional[SABRParams] = None

    def hagan_implied_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> float:
        """Hagan's closed-form SABR implied volatility approximation.

        Args:
            F: Forward price.
            K: Strike price.
            T: Time to expiry in years.
            alpha: SABR alpha parameter.
            rho: SABR rho parameter.
            nu: SABR nu (vol-of-vol) parameter.

        Returns:
            Black implied volatility.
        """
        beta = self.beta

        if abs(F - K) < 1e-12:
            # ATM formula
            FK_mid = F
            logFK = 0.0
            term1 = alpha / (FK_mid ** (1 - beta))
            term2 = 1.0 + T * (
                ((1 - beta) ** 2 / 24) * alpha ** 2 / (FK_mid ** (2 * (1 - beta)))
                + (rho * beta * nu * alpha) / (4 * FK_mid ** (1 - beta))
                + (2 - 3 * rho ** 2) * nu ** 2 / 24
            )
            return max(term1 * term2, 1e-8)

        FK_mid = (F * K) ** ((1 - beta) / 2)
        logFK = np.log(F / K)

        # z and x(z)
        z = (nu / alpha) * FK_mid * logFK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        if abs(x_z) < 1e-12:
            zeta_ratio = 1.0
        else:
            zeta_ratio = z / x_z

        # Denominator correction
        denom = FK_mid * (
            1
            + ((1 - beta) ** 2 / 24) * logFK ** 2
            + ((1 - beta) ** 4 / 1920) * logFK ** 4
        )

        # Numerator correction
        numer_corr = 1.0 + T * (
            ((1 - beta) ** 2 / 24) * alpha ** 2 / (FK_mid ** 2)
            + (rho * beta * nu * alpha) / (4 * FK_mid)
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        )

        sigma = (alpha / denom) * zeta_ratio * numer_corr
        return max(float(sigma), 1e-8)

    def implied_vol_surface(
        self,
        F: float,
        strikes: np.ndarray,
        T: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> np.ndarray:
        """Compute implied volatilities for multiple strikes."""
        return np.array([
            self.hagan_implied_vol(F, K, T, alpha, rho, nu)
            for K in strikes
        ])

    def calibrate(
        self,
        F: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        alpha0: float = 0.2,
        rho0: float = -0.3,
        nu0: float = 0.4,
    ) -> SABRCalibrationResult:
        """Calibrate SABR parameters to market implied vols using Nelder-Mead.

        Args:
            F: Forward price.
            strikes: Array of strike prices.
            market_vols: Array of market implied volatilities.
            T: Time to expiry.
            alpha0, rho0, nu0: Initial parameter guesses.

        Returns:
            SABRCalibrationResult with calibrated parameters.
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        market_vols = np.asarray(market_vols, dtype=np.float64)

        def objective(params):
            a, r, n = params
            # Constraints
            if a <= 0 or n <= 0 or abs(r) >= 1:
                return 1e6
            try:
                model_vols = self.implied_vol_surface(F, strikes, T, a, r, n)
                return np.sum((model_vols - market_vols) ** 2)
            except Exception:
                return 1e6

        result = minimize(
            objective,
            [alpha0, rho0, nu0],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
        )

        if result.success:
            alpha_cal, rho_cal, nu_cal = result.x
            params = SABRParams(
                alpha=float(alpha_cal),
                beta=self.beta,
                rho=float(np.clip(rho_cal, -0.999, 0.999)),
                nu=float(max(nu_cal, 1e-6)),
            )
            model_vols = self.implied_vol_surface(
                F, strikes, T, params.alpha, params.rho, params.nu
            )
            rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))
        else:
            params = SABRParams(alpha=alpha0, beta=self.beta, rho=rho0, nu=nu0)
            rmse = float("inf")

        self._params = params
        cal_result = SABRCalibrationResult(
            params=params,
            rmse=rmse,
            n_strikes=len(strikes),
            success=result.success,
        )
        logger.info(
            "SABR calibration: alpha=%.4f, rho=%.3f, nu=%.4f, RMSE=%.6f",
            params.alpha, params.rho, params.nu, rmse,
        )
        return cal_result

    @property
    def params(self) -> Optional[SABRParams]:
        return self._params
