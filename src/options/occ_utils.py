"""
OCC Symbol Parsing Utilities
==============================

Centralized OCC (Options Clearing Corporation) symbol parsing used across
the options trading engine.  Every module that needs to extract the
underlying, expiry, option type, or strike from an OCC symbol should use
``parse_occ_symbol()`` instead of inline character loops.

OCC Format:   AAPL260320P00230000
              ^^^^------           underlying (1-6 alpha chars)
                  ^^^^^^           YYMMDD expiration
                        ^          'P' (put) or 'C' (call)
                         ^^^^^^^^  strike × 1000 (8 digits, zero-padded)

Functions:
    parse_occ_symbol   — full parse returning a dict
    compute_option_delta — approximate B-S delta from OCC + underlying price
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Pre-compiled regex for OCC parsing.
# Underlying: 1-6 uppercase letters, date: 6 digits, type: P|C, strike: 8 digits
_OCC_RE = re.compile(
    r'^(?P<underlying>[A-Z]{1,6})'
    r'(?P<date>\d{6})'
    r'(?P<type>[PC])'
    r'(?P<strike>\d{8})$'
)


def parse_occ_symbol(occ: str) -> Optional[dict]:
    """Parse an OCC option symbol into its components.

    Args:
        occ: OCC symbol string, e.g. ``"SPY260320P00550000"``.

    Returns:
        A dict with keys ``underlying``, ``expiry_date`` (``datetime.date``),
        ``option_type`` (``"P"`` or ``"C"``), and ``strike`` (float, in
        dollars).  Returns ``None`` if *occ* cannot be parsed.

    Examples:
        >>> parse_occ_symbol("AAPL260320P00230000")
        {'underlying': 'AAPL', 'expiry_date': date(2026,3,20),
         'option_type': 'P', 'strike': 230.0}
        >>> parse_occ_symbol("A260620C00150000")
        {'underlying': 'A', 'expiry_date': date(2026,6,20),
         'option_type': 'C', 'strike': 150.0}
    """
    if not occ or not isinstance(occ, str):
        return None

    occ = occ.strip().upper()
    m = _OCC_RE.match(occ)
    if m is None:
        # Fallback: walk until first digit to find underlying boundary
        try:
            idx = 0
            for ch in occ:
                if ch.isdigit():
                    break
                idx += 1
            if idx == 0 or idx >= len(occ) - 14:
                return None
            underlying = occ[:idx]
            rest = occ[idx:]
            if len(rest) < 15:
                return None
            date_str = rest[:6]
            opt_type = rest[6]
            strike_str = rest[7:15]
            yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
            expiry = date(2000 + yy, mm, dd)
            strike = int(strike_str) / 1000.0
            if opt_type not in ('P', 'C'):
                return None
            return {
                'underlying': underlying,
                'expiry_date': expiry,
                'option_type': opt_type,
                'strike': strike,
            }
        except (ValueError, IndexError):
            return None

    try:
        ds = m.group('date')
        yy, mm, dd = int(ds[:2]), int(ds[2:4]), int(ds[4:6])
        expiry = date(2000 + yy, mm, dd)
    except (ValueError, IndexError):
        return None

    strike = int(m.group('strike')) / 1000.0

    return {
        'underlying': m.group('underlying'),
        'expiry_date': expiry,
        'option_type': m.group('type'),
        'strike': strike,
    }


def compute_option_delta(
    occ_symbol: str,
    underlying_price: float,
    risk_free_rate: float = 0.05,
    implied_vol: Optional[float] = None,
) -> float:
    """Compute approximate Black-Scholes delta for an option.

    Parses the OCC symbol to extract strike / expiry / type, then uses
    the standard BS delta formula.  The result is **per-share** delta
    (range roughly -1.0 to +1.0).  Multiply by ``qty * 100`` to get
    portfolio-level share-equivalent delta.

    Falls back to a simple moneyness-based estimate when the BS calc
    fails (e.g. expired option, bad inputs).

    Args:
        occ_symbol: OCC symbol string.
        underlying_price: Current price of the underlying.
        risk_free_rate: Annual risk-free rate (default 5 %).
        implied_vol: Annualised implied volatility.  If ``None``, a
            rough estimate is derived from the ATM-ness of the option.

    Returns:
        Per-share delta (positive for calls, negative for puts).
        Falls back to ±0.50 simple estimate on error.
    """
    parsed = parse_occ_symbol(occ_symbol)
    if parsed is None or underlying_price <= 0:
        # Cannot parse — return crude ±0.50 fallback
        is_put = 'P' in occ_symbol.upper()
        return -0.50 if is_put else 0.50

    strike = parsed['strike']
    expiry = parsed['expiry_date']
    opt_type = parsed['option_type']

    # DTE
    today = date.today()
    dte = (expiry - today).days
    if dte <= 0:
        # Expired or expiring today — intrinsic-only delta
        if opt_type == 'C':
            return 1.0 if underlying_price > strike else 0.0
        else:
            return -1.0 if underlying_price < strike else 0.0

    T = dte / 365.0

    # Implied vol estimate when not provided
    if implied_vol is None or implied_vol <= 0:
        # Rough estimate: ~25% baseline, scale up for far OTM
        moneyness = underlying_price / strike if strike > 0 else 1.0
        implied_vol = 0.25 + 0.10 * abs(math.log(max(moneyness, 0.01)))
        implied_vol = max(0.10, min(implied_vol, 1.50))  # clamp

    try:
        sigma = implied_vol
        S = underlying_price
        K = strike
        r = risk_free_rate

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if opt_type == 'C':
            delta = float(norm.cdf(d1))
        else:
            delta = float(norm.cdf(d1) - 1.0)   # negative for puts

        return delta

    except Exception as exc:
        logger.debug(f"BS delta calc failed for {occ_symbol}: {exc}")
        # Simple fallback based on moneyness
        try:
            moneyness = underlying_price / strike
            if opt_type == 'C':
                if moneyness > 1.05:
                    return 0.80
                elif moneyness > 0.95:
                    return 0.50
                else:
                    return 0.20
            else:
                if moneyness < 0.95:
                    return -0.80
                elif moneyness < 1.05:
                    return -0.50
                else:
                    return -0.20
        except Exception:
            return -0.50 if opt_type == 'P' else 0.50


def smart_limit_price(
    bid: float,
    ask: float,
    side: str,
    aggression: float = 0.30,
) -> float:
    """Compute a smart limit price that leans toward the favorable side.

    For **buys** the favorable side is the bid (lower), so the limit is
    set at ``bid + aggression * (ask - bid)`` — usually 30 % into the
    spread from the bid.

    For **sells** the favorable side is the ask (higher), so the limit is
    set at ``ask - aggression * (ask - bid)`` — usually 30 % into the
    spread from the ask.

    Falls back to mid-price when inputs are invalid.

    Args:
        bid: Best bid price.
        ask: Best ask price.
        side: ``"buy"`` or ``"sell"``.
        aggression: How far into the spread to lean (0 = favorable edge,
            0.5 = mid, 1.0 = aggressive / crossing).

    Returns:
        Suggested limit price rounded to 2 decimal places.
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        # Bad quotes — use mid or whichever is available
        mid = max(bid, ask)
        if mid <= 0:
            mid = 0.01
        return round(mid, 2)

    spread = ask - bid
    aggression = max(0.0, min(1.0, aggression))

    if side.lower() in ("buy", "b"):
        price = bid + aggression * spread
    else:
        price = ask - aggression * spread

    return round(max(0.01, price), 2)
