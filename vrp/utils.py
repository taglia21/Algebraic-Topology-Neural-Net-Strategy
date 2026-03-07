"""
vrp/utils.py
============
Shared utilities: logging, Black-Scholes greeks, date helpers.

The greeks module implements analytical Black-Scholes for European options.
SPX options are European-style and cash-settled, so BS is appropriate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Fast normal distribution (replaces scipy.stats.norm for 10x speedup)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT2PI = 1.0 / _SQRT2PI


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc (C-level, no scipy overhead)."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _norm_pdf(x: float) -> float:
    """Standard normal PDF — pure math, no scipy."""
    return _INV_SQRT2PI * math.exp(-0.5 * x * x)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "vrp", level: str = "INFO") -> logging.Logger:
    """Create a configured logger with clean formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------

@dataclass
class Greeks:
    """Option greeks from Black-Scholes model."""
    delta: float
    gamma: float
    theta: float  # per day
    vega: float   # per 1% IV move
    rho: float
    iv: float     # implied volatility used


def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in Black-Scholes formula.

    Parameters
    ----------
    S : Underlying price
    K : Strike price
    T : Time to expiration in years
    r : Risk-free rate (annualized)
    sigma : Volatility (annualized)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put option price via Black-Scholes.

    Parameters
    ----------
    S : Current underlying price
    K : Strike price
    T : Time to expiration in years
    r : Risk-free rate
    sigma : Implied volatility
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call option price via Black-Scholes."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "put",
) -> Greeks:
    """Calculate full greeks for a European option.

    Parameters
    ----------
    S : Underlying price
    K : Strike price
    T : Time to expiration in years
    r : Risk-free rate
    sigma : Implied volatility
    option_type : "put" or "call"
    """
    if T <= 1e-10 or sigma <= 1e-10:
        intrinsic = max(K - S, 0) if option_type == "put" else max(S - K, 0)
        delta = -1.0 if (option_type == "put" and S < K) else (1.0 if option_type == "call" and S > K else 0.0)
        return Greeks(delta=delta, gamma=0, theta=0, vega=0, rho=0, iv=sigma)

    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)

    # Common terms — use fast pure-math CDF/PDF
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    npd1 = _norm_pdf(d1)
    discount = math.exp(-r * T)

    if option_type == "put":
        delta = nd1 - 1.0  # negative for puts
        rho = -K * T * discount * _norm_cdf(-d2) / 100.0
    else:
        delta = nd1
        rho = K * T * discount * nd2 / 100.0

    gamma = npd1 / (S * sigma * sqrt_T)

    # Theta: per calendar day (divide annual by 365)
    theta_term1 = -(S * npd1 * sigma) / (2 * sqrt_T)
    if option_type == "put":
        theta_term2 = r * K * discount * _norm_cdf(-d2)
        theta = (theta_term1 + theta_term2) / 365.0
    else:
        theta_term2 = -r * K * discount * nd2
        theta = (theta_term1 + theta_term2) / 365.0

    # Vega: per 1% move in IV
    vega = S * npd1 * sqrt_T / 100.0

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho, iv=sigma)


def implied_vol(
    price: float, S: float, K: float, T: float, r: float,
    option_type: str = "put",
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """Newton-Raphson implied volatility solver.

    Parameters
    ----------
    price : Market price of the option
    S, K, T, r : Standard BS parameters
    option_type : "put" or "call"
    tol : Convergence tolerance
    max_iter : Maximum iterations
    """
    if price <= 0:
        return 0.0

    # Initial guess from Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / T) * price / S

    pricer = bs_put_price if option_type == "put" else bs_call_price

    for _ in range(max_iter):
        calc_price = pricer(S, K, T, r, sigma)
        diff = calc_price - price

        if abs(diff) < tol:
            return sigma

        # Vega for Newton step
        d1 = bs_d1(S, K, T, r, sigma)
        vega = S * _norm_pdf(d1) * math.sqrt(T)

        if vega < 1e-12:
            break

        sigma -= diff / vega
        sigma = max(sigma, 0.001)  # floor at 0.1%
        sigma = min(sigma, 5.0)    # cap at 500%

    return sigma


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def trading_days_between(start: date, end: date) -> int:
    """Count trading days between two dates (approximate, excludes weekends)."""
    if start >= end:
        return 0
    days = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            days += 1
    return days


def next_monthly_expiry(from_date: date) -> date:
    """Find the next monthly options expiry (3rd Friday of the month).

    SPX standard monthly options expire on the 3rd Friday.
    """
    year, month = from_date.year, from_date.month

    # Find 3rd Friday of current month
    first_day = date(year, month, 1)
    # Day of week: 0=Monday, 4=Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    third_friday = first_friday + timedelta(weeks=2)

    if third_friday > from_date:
        return third_friday

    # Move to next month
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1

    first_day = date(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    return first_friday + timedelta(weeks=2)


def dte(expiry: date, as_of: Optional[date] = None) -> int:
    """Days to expiration from a given date."""
    as_of = as_of or date.today()
    return (expiry - as_of).days


def years_to_expiry(expiry: date, as_of: Optional[date] = None) -> float:
    """Time to expiration in years (for BS calculations)."""
    return dte(expiry, as_of) / 365.0
