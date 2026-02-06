"""
Dupire Local Volatility Model
===============================
Extracts the local volatility surface from market option prices using
Dupire's formula (1994):

    σ_local²(K, T) = (∂C/∂T + (r-q)K·∂C/∂K + qC) / (0.5 K² ∂²C/∂K²)

The local vol surface is a deterministic function σ(S, t) that is
consistent with all observed European option prices.

Alternatively, from the implied volatility surface σ_BS(K, T):

    σ_local²(K, T) = [∂w/∂T] / [1 - y/w · ∂w/∂y + 0.25(-0.25 - 1/w + y²/w²)(∂w/∂y)² + 0.5 · ∂²w/∂y²]

where w = σ_BS² · T (total implied variance), y = ln(K/F).

Implementation:
- Constructs implied vol surface from market quotes
- Computes local vol via Dupire's formula with numerical derivatives
- Interpolation using cubic spline on (K, T) grid
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class VolSurfacePoint:
    """A single point on the volatility surface."""
    strike: float
    expiry: float  # years
    implied_vol: float
    local_vol: float = 0.0
    market_price: float = 0.0


@dataclass
class LocalVolSurface:
    """The complete local volatility surface."""
    strikes: np.ndarray
    expiries: np.ndarray
    local_vols: np.ndarray   # shape (len(expiries), len(strikes))
    implied_vols: np.ndarray  # shape (len(expiries), len(strikes))
    spot: float
    risk_free_rate: float


class DupireLocalVol:
    """
    Dupire local volatility surface construction.
    
    Workflow:
    1. Input implied vol surface data (strike, expiry, IV)
    2. Fit smooth interpolated surface
    3. Apply Dupire's formula for local vol at each grid point
    4. Return callable local vol surface
    """

    def __init__(self, risk_free_rate: float = 0.045, dividend_yield: float = 0.0):
        self.rf = risk_free_rate
        self.q = dividend_yield
        self._surface: Optional[LocalVolSurface] = None
        self._iv_interp: Optional[RectBivariateSpline] = None
        self._lv_interp: Optional[RectBivariateSpline] = None

    def build_surface(
        self,
        spot: float,
        market_data: List[Dict],
        strike_grid: Optional[np.ndarray] = None,
        expiry_grid: Optional[np.ndarray] = None,
    ) -> LocalVolSurface:
        """
        Build local vol surface from market implied vol data.
        
        market_data: list of dicts with keys {strike, expiry, implied_vol}
                     or {strike, expiry, price, is_call}
        """
        # Extract unique strikes and expiries
        strikes_raw = sorted(set(d["strike"] for d in market_data))
        expiries_raw = sorted(set(d["expiry"] for d in market_data))

        if len(strikes_raw) < 3 or len(expiries_raw) < 2:
            raise ValueError(
                f"Need at least 3 strikes and 2 expiries, got {len(strikes_raw)} and {len(expiries_raw)}"
            )

        # Build IV matrix from market data
        iv_dict: Dict[Tuple[float, float], float] = {}
        for d in market_data:
            K, T = d["strike"], d["expiry"]
            if "implied_vol" in d:
                iv_dict[(K, T)] = d["implied_vol"]
            elif "price" in d:
                iv_dict[(K, T)] = self._price_to_iv(
                    spot, K, T, d["price"], d.get("is_call", True)
                )

        # Create regular grids
        if strike_grid is None:
            strike_grid = np.linspace(
                min(strikes_raw) * 0.95, max(strikes_raw) * 1.05, max(len(strikes_raw), 30)
            )
        if expiry_grid is None:
            expiry_grid = np.linspace(
                max(min(expiries_raw), 0.01), max(expiries_raw), max(len(expiries_raw), 15)
            )

        # Interpolate IV onto regular grid
        # First build a rough IV matrix
        iv_matrix = np.zeros((len(expiries_raw), len(strikes_raw)))
        for i, T in enumerate(expiries_raw):
            for j, K in enumerate(strikes_raw):
                key = (K, T)
                if key in iv_dict:
                    iv_matrix[i, j] = iv_dict[key]
                else:
                    # Interpolate from nearest
                    iv_matrix[i, j] = self._interpolate_iv(K, T, iv_dict, spot)

        # Ensure no zeros
        mean_iv = np.mean(iv_matrix[iv_matrix > 0]) if np.any(iv_matrix > 0) else 0.2
        iv_matrix[iv_matrix <= 0] = mean_iv

        # Fit smooth bivariate spline
        try:
            self._iv_interp = RectBivariateSpline(
                np.array(expiries_raw), np.array(strikes_raw), iv_matrix, kx=3, ky=3
            )
        except Exception:
            # Fall back to linear
            self._iv_interp = RectBivariateSpline(
                np.array(expiries_raw), np.array(strikes_raw), iv_matrix, kx=1, ky=1
            )

        # Evaluate IV on fine grid
        iv_fine = self._iv_interp(expiry_grid, strike_grid)
        iv_fine = np.clip(iv_fine, 0.01, 5.0)

        # Apply Dupire's formula for local vol
        local_vols = self._compute_local_vol(spot, strike_grid, expiry_grid, iv_fine)

        self._surface = LocalVolSurface(
            strikes=strike_grid,
            expiries=expiry_grid,
            local_vols=local_vols,
            implied_vols=iv_fine,
            spot=spot,
            risk_free_rate=self.rf,
        )

        # Build local vol interpolator
        self._lv_interp = RectBivariateSpline(
            expiry_grid, strike_grid, local_vols, kx=3, ky=3
        )

        logger.info(
            f"Dupire surface built: {len(expiry_grid)} expiries × {len(strike_grid)} strikes, "
            f"local vol range [{local_vols.min():.2%}, {local_vols.max():.2%}]"
        )
        return self._surface

    def _compute_local_vol(
        self,
        S: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
        iv_surface: np.ndarray,
    ) -> np.ndarray:
        """
        Dupire's formula from implied vol surface.
        
        Using total variance w = σ²T:
        
        σ²_local = ∂w/∂T / denominator
        
        denominator = 1 - (y/w)(∂w/∂y) + 0.25(-0.25 - 1/w + y²/w²)(∂w/∂y)² + 0.5(∂²w/∂y²)
        
        where y = ln(K/F), F = S·exp((r-q)T)
        """
        r = self.rf
        q = self.q
        n_T = len(expiries)
        n_K = len(strikes)
        local_vols = np.full((n_T, n_K), 0.2)

        # Total variance surface
        w = iv_surface ** 2 * expiries[:, None]  # shape (n_T, n_K)

        for i in range(1, n_T - 1):
            T = expiries[i]
            dT = expiries[i + 1] - expiries[i - 1]
            F = S * np.exp((r - q) * T)

            for j in range(1, n_K - 1):
                K = strikes[j]
                dK = strikes[j + 1] - strikes[j - 1]

                y = np.log(K / F)
                w_val = w[i, j]

                if w_val < 1e-10:
                    local_vols[i, j] = iv_surface[i, j]
                    continue

                # Numerical derivatives
                dw_dT = (w[i + 1, j] - w[i - 1, j]) / dT
                dw_dy = (w[i, j + 1] - w[i, j - 1]) / (np.log(strikes[j + 1] / F) - np.log(strikes[j - 1] / F) + 1e-12)
                d2w_dy2 = (w[i, j + 1] - 2 * w[i, j] + w[i, j - 1]) / ((np.log(strikes[j + 1] / F) - np.log(strikes[j] / F)) ** 2 + 1e-12)

                # Dupire denominator
                denom = (
                    1.0
                    - (y / w_val) * dw_dy
                    + 0.25 * (-0.25 - 1.0 / w_val + y ** 2 / w_val ** 2) * dw_dy ** 2
                    + 0.5 * d2w_dy2
                )

                if denom <= 0 or dw_dT <= 0:
                    local_vols[i, j] = iv_surface[i, j]
                else:
                    lv2 = dw_dT / denom
                    if lv2 > 0:
                        local_vols[i, j] = np.sqrt(lv2 / T) if T > 0 else iv_surface[i, j]
                    else:
                        local_vols[i, j] = iv_surface[i, j]

        # Fill boundaries
        local_vols[0, :] = local_vols[1, :]
        local_vols[-1, :] = local_vols[-2, :]
        local_vols[:, 0] = local_vols[:, 1]
        local_vols[:, -1] = local_vols[:, -2]

        # Clip to reasonable range
        local_vols = np.clip(local_vols, 0.01, 5.0)

        return local_vols

    def get_local_vol(self, K: float, T: float) -> float:
        """Get interpolated local vol at (K, T)."""
        if self._lv_interp is None:
            raise ValueError("Surface not built yet. Call build_surface() first.")
        val = self._lv_interp(T, K)[0, 0]
        return float(np.clip(val, 0.01, 5.0))

    def get_implied_vol(self, K: float, T: float) -> float:
        """Get interpolated implied vol at (K, T)."""
        if self._iv_interp is None:
            raise ValueError("Surface not built yet. Call build_surface() first.")
        val = self._iv_interp(T, K)[0, 0]
        return float(np.clip(val, 0.01, 5.0))

    def _price_to_iv(
        self, S: float, K: float, T: float, price: float, is_call: bool
    ) -> float:
        """BSM implied vol via bisection."""
        lo, hi = 0.01, 3.0
        for _ in range(100):
            mid = (lo + hi) / 2
            p = self._bsm(S, K, T, mid, is_call)
            if abs(p - price) < 1e-8:
                return mid
            if p > price:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    def _bsm(self, S: float, K: float, T: float, sigma: float, is_call: bool) -> float:
        """BSM price."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)
        r = self.rf
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def _interpolate_iv(
        K: float, T: float, iv_dict: Dict[Tuple[float, float], float], spot: float
    ) -> float:
        """Simple nearest-neighbor interpolation fallback."""
        if not iv_dict:
            return 0.2
        best_dist = float("inf")
        best_iv = 0.2
        for (k, t), iv in iv_dict.items():
            dist = abs(k / spot - K / spot) + abs(t - T)
            if dist < best_dist:
                best_dist = dist
                best_iv = iv
        return best_iv

    def generate_synthetic_surface(
        self, spot: float, base_vol: float = 0.25
    ) -> LocalVolSurface:
        """
        Generate a synthetic vol surface for testing.
        Uses a parametric skew model: σ(K,T) = base * (1 + skew*moneyness + smile*moneyness²) * term_adj
        """
        strikes = np.linspace(spot * 0.7, spot * 1.3, 40)
        expiries = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

        market_data = []
        for T in expiries:
            for K in strikes:
                moneyness = np.log(K / spot)
                skew = -0.3  # negative skew for equities
                smile = 0.5
                term_factor = 1.0 + 0.1 * np.sqrt(max(T, 0.01))
                iv = base_vol * (1 + skew * moneyness + smile * moneyness ** 2) * term_factor
                iv = max(iv, 0.05)
                market_data.append({"strike": K, "expiry": T, "implied_vol": iv})

        return self.build_surface(spot, market_data, strikes, expiries)
