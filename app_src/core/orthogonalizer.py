"""
core/orthogonalizer.py
======================
ORIA-inspired factor orthogonalization layer.

Projects raw strategy weight vectors onto the space orthogonal to known
risk factors, isolating pure alpha from systematic beta / factor exposure.

Mathematics (from Joshua Aalampour's ORIA Part 1):
    ũ_t^s = arg min  ½ (u - u_t^s)^T  Ω_t  (u - u_t^s)
            subject to   F_t^T u = 0

Closed-form solution via projection matrix:
    P = I - F (F^T Ω^{-1} F)^{-1} F^T Ω^{-1}
    ũ_t^s = P · u_t^s

When Ω = I (identity, i.e. equal risk weighting):
    P = I - F (F^T F)^{-1} F^T

This removes all linear exposure to the columns of F from the signal vector.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_factor_matrix(
    returns_df: pd.DataFrame,
    symbols: List[str],
    factors: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """Build the factor loading matrix F from recent return data.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, columns = symbols.
    symbols : list
        Ordered list of symbols (defines row order).
    factors : list, optional
        Column names in returns_df to use as factors.
        Default: ['SPY'] (market beta only).

    Returns
    -------
    np.ndarray or None
        Shape (n_symbols, n_factors) — each column is a factor loading vector.
    """
    if returns_df is None or returns_df.empty:
        return None

    if factors is None:
        factors = ["SPY"]

    # Only use factors that exist in the data
    available = [f for f in factors if f in returns_df.columns]
    if not available:
        return None

    n = len(symbols)
    k = len(available)
    F = np.zeros((n, k))

    for j, factor_name in enumerate(available):
        factor_returns = returns_df[factor_name].values
        for i, sym in enumerate(symbols):
            if sym == factor_name:
                F[i, j] = 1.0
            elif sym in returns_df.columns:
                sym_returns = returns_df[sym].values
                # Simple OLS beta: β = cov(r_i, f_j) / var(f_j)
                valid = ~(np.isnan(sym_returns) | np.isnan(factor_returns))
                if valid.sum() > 10:
                    cov_mat = np.cov(sym_returns[valid], factor_returns[valid])
                    var_f = cov_mat[1, 1]
                    if var_f > 1e-12:
                        F[i, j] = cov_mat[0, 1] / var_f
    return F


def compute_projection_matrix(
    F: np.ndarray,
    omega: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the orthogonal projection matrix P.

    P = I - F (F^T Ω^{-1} F)^{-1} F^T Ω^{-1}

    When omega is None (identity), simplifies to:
    P = I - F (F^T F)^{-1} F^T

    Parameters
    ----------
    F : np.ndarray
        Factor loading matrix, shape (n, k).
    omega : np.ndarray, optional
        Risk weighting matrix, shape (n, n). Default: identity.

    Returns
    -------
    np.ndarray
        Projection matrix P, shape (n, n).
    """
    n = F.shape[0]

    if omega is None:
        # Simple case: P = I - F (F^T F)^{-1} F^T
        FtF = F.T @ F
        try:
            FtF_inv = np.linalg.inv(FtF + 1e-8 * np.eye(FtF.shape[0]))
        except np.linalg.LinAlgError:
            logger.warning("Factor matrix singular, returning identity")
            return np.eye(n)
        P = np.eye(n) - F @ FtF_inv @ F.T
    else:
        # General case with risk weighting
        try:
            omega_inv = np.linalg.inv(omega + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            omega_inv = np.eye(n)

        Ft_oi = F.T @ omega_inv
        Ft_oi_F = Ft_oi @ F
        try:
            Ft_oi_F_inv = np.linalg.inv(Ft_oi_F + 1e-8 * np.eye(Ft_oi_F.shape[0]))
        except np.linalg.LinAlgError:
            logger.warning("Weighted factor matrix singular, returning identity")
            return np.eye(n)
        P = np.eye(n) - F @ Ft_oi_F_inv @ Ft_oi

    return P


class SignalOrthogonalizer:
    """ORIA-style signal orthogonalization layer.

    Removes systematic factor exposure from strategy signal vectors,
    isolating the idiosyncratic (alpha) component.

    Parameters
    ----------
    factor_symbols : list
        Symbols to use as risk factors for neutralization.
        Default: ['SPY'] (market-only neutralization).
        Recommended: ['SPY', 'IWM', 'GLD', 'TLT'] for multi-factor.
    lookback : int
        Number of return observations for beta estimation.
    use_covariance_weighting : bool
        If True, use Ω = Σ (covariance matrix) for risk-weighted projection.
        If False, use Ω = I (simple projection).
    """

    def __init__(
        self,
        factor_symbols: Optional[List[str]] = None,
        lookback: int = 60,
        use_covariance_weighting: bool = False,
    ):
        self.factor_symbols = factor_symbols or ["SPY"]
        self.lookback = lookback
        self.use_covariance_weighting = use_covariance_weighting
        self._last_P: Optional[np.ndarray] = None
        self._last_factors_used: List[str] = []

    def orthogonalize_signals(
        self,
        signals_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        covariance_matrix: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Orthogonalize strategy signals against risk factors.

        Parameters
        ----------
        signals_df : pd.DataFrame
            Columns: ticker, direction, strength (raw strategy output).
        returns_df : pd.DataFrame
            Recent daily returns, columns = symbols. Used to estimate betas.
        covariance_matrix : np.ndarray, optional
            If provided and use_covariance_weighting=True, used as Ω.

        Returns
        -------
        pd.DataFrame
            Same structure as input but with strengths adjusted to remove
            factor exposure. direction may flip if alpha is opposite to beta.
        """
        if signals_df.empty or returns_df is None or returns_df.empty:
            return signals_df.copy()

        tickers = sorted(signals_df["ticker"].unique())
        n = len(tickers)

        if n < 2:
            return signals_df.copy()

        # Build raw signal weight vector
        ticker_idx = {t: i for i, t in enumerate(tickers)}
        u_raw = np.zeros(n)
        directions = {}

        for _, row in signals_df.iterrows():
            t = row["ticker"]
            if t in ticker_idx:
                sign = 1.0 if row["direction"] == "LONG" else -1.0
                u_raw[ticker_idx[t]] = sign * float(row["strength"])
                directions[t] = row["direction"]

        # Build factor matrix
        # Use last `lookback` rows of returns
        recent = returns_df.tail(self.lookback)
        F = _build_factor_matrix(recent, tickers, self.factor_symbols)

        if F is None or F.shape[1] == 0:
            logger.info("No factor data available, skipping orthogonalization")
            return signals_df.copy()

        self._last_factors_used = [
            f for f in self.factor_symbols if f in returns_df.columns
        ]

        # Compute projection matrix
        omega = None
        if self.use_covariance_weighting and covariance_matrix is not None:
            # Extract sub-matrix for our tickers
            if covariance_matrix.shape[0] >= n:
                omega = covariance_matrix[:n, :n]

        P = compute_projection_matrix(F, omega)
        self._last_P = P

        # Project signal vector onto factor-orthogonal space
        u_ortho = P @ u_raw

        # Compute factor exposure removed
        factor_exposure = u_raw - u_ortho
        total_removed = np.abs(factor_exposure).sum()
        logger.info(
            "Orthogonalization: removed %.3f total factor exposure across %d factors (%s)",
            total_removed, len(self._last_factors_used), self._last_factors_used,
        )

        # Rebuild signals DataFrame
        records = []
        for _, row in signals_df.iterrows():
            t = row["ticker"]
            if t in ticker_idx:
                ortho_val = u_ortho[ticker_idx[t]]
                new_strength = abs(ortho_val)
                # Direction from the sign of orthogonalized value
                if ortho_val > 0.001:
                    new_dir = "LONG"
                elif ortho_val < -0.001:
                    new_dir = "SHORT"
                else:
                    new_dir = "NEUTRAL"

                records.append({
                    "ticker": t,
                    "direction": new_dir,
                    "strength": round(min(1.0, new_strength), 6),
                    "raw_direction": row["direction"],
                    "raw_strength": float(row["strength"]),
                    "factor_exposure_removed": round(abs(factor_exposure[ticker_idx[t]]), 6),
                    "timestamp": row.get("timestamp"),
                })
            else:
                records.append(dict(row))

        result = pd.DataFrame(records)
        return result

    def get_factor_exposures(self) -> Dict[str, float]:
        """Return the last computed factor exposures for diagnostics."""
        return {
            "factors_used": self._last_factors_used,
            "projection_rank": (
                int(np.linalg.matrix_rank(self._last_P))
                if self._last_P is not None else 0
            ),
        }
