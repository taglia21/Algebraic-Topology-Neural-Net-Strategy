"""
core/atr_brackets.py
====================
ATR-based dynamic take-profit / stop-loss bracket calculator.

Instead of flat percentages (6%/3%), scales brackets to each ticker's
actual volatility using Average True Range. This means:
- High-vol stocks (TSLA, NVDA) get wider brackets → not stopped out on noise
- Low-vol stocks (GLD, TLT) get tighter brackets → faster profit capture

Default: 1.5x ATR stop-loss, 2.0x ATR take-profit (adjustable per regime).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime multipliers for bracket width
# Wider brackets = swing-trade friendly (avoids PDT on margin accounts <$25K)
# Positions should survive intraday noise and hold 1-3 days for full P&L capture
_REGIME_SL_MULT = {
    "NORMAL": 2.5,    # Widened from 1.5 → 2.5 for swing-trade holds
    "STRESSED": 2.0,  # Widened from 1.2 → 2.0
    "CRASH": 1.5,     # Widened from 1.0 → 1.5
}
_REGIME_TP_MULT = {
    "NORMAL": 3.5,    # Widened from 2.0 → 3.5 for larger profit targets
    "STRESSED": 3.0,  # Widened from 1.8 → 3.0
    "CRASH": 2.5,     # Widened from 1.5 → 2.5
}

# Absolute floor/ceiling for bracket percentages
# Widened floors to ensure positions don't trigger same-day (PDT avoidance)
_MIN_SL_PCT = 0.02    # 2% minimum stop (was 0.8% — too tight, triggered same-day)
_MAX_SL_PCT = 0.08    # 8% maximum stop (was 5%)
_MIN_TP_PCT = 0.03    # 3% minimum TP (was 1.2% — too tight, triggered same-day)
_MAX_TP_PCT = 0.12    # 12% maximum TP (was 8%)


@dataclass
class BracketLevels:
    """Computed bracket levels for an order."""
    ticker: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    atr_value: float
    atr_pct: float           # ATR as % of price
    regime: str
    direction: str           # LONG or SHORT


def compute_atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> float:
    """Compute Average True Range for the most recent period.

    Parameters
    ----------
    highs : pd.Series
        High prices.
    lows : pd.Series
        Low prices.
    closes : pd.Series
        Close prices.
    period : int
        ATR lookback period (default 14).

    Returns
    -------
    float
        Current ATR value.
    """
    if len(closes) < period + 1:
        # Fallback: use simple range of last few bars
        if len(highs) > 1:
            return float((highs - lows).tail(min(period, len(highs))).mean())
        return float(closes.iloc[-1] * 0.02)  # 2% fallback

    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else float(closes.iloc[-1] * 0.02)


def compute_atr_from_closes(
    closes: pd.Series,
    period: int = 14,
) -> float:
    """Estimate ATR from close prices only (when H/L not available).

    Uses absolute daily returns as a proxy for true range.
    """
    if len(closes) < period + 1:
        return float(closes.iloc[-1] * 0.02)

    daily_range = closes.diff().abs()
    atr_est = daily_range.rolling(period).mean().iloc[-1]

    # Scale up slightly since close-to-close underestimates true range
    return float(atr_est * 1.3) if not np.isnan(atr_est) else float(closes.iloc[-1] * 0.02)


def calculate_brackets(
    ticker: str,
    entry_price: float,
    closes: pd.Series,
    regime: str = "NORMAL",
    direction: str = "LONG",
    atr_period: int = 14,
    sl_atr_mult: Optional[float] = None,
    tp_atr_mult: Optional[float] = None,
    highs: Optional[pd.Series] = None,
    lows: Optional[pd.Series] = None,
) -> BracketLevels:
    """Calculate ATR-based bracket levels for an order.

    Parameters
    ----------
    ticker : str
        Stock symbol.
    entry_price : float
        Expected entry price.
    closes : pd.Series
        Historical close prices.
    regime : str
        Market regime (NORMAL, STRESSED, CRASH).
    direction : str
        LONG or SHORT.
    atr_period : int
        ATR lookback (default 14).
    sl_atr_mult : float, optional
        Override stop-loss ATR multiplier.
    tp_atr_mult : float, optional
        Override take-profit ATR multiplier.
    highs, lows : pd.Series, optional
        High/Low prices for true ATR. If None, estimates from closes.

    Returns
    -------
    BracketLevels
        Complete bracket specification.
    """
    # Compute ATR
    if highs is not None and lows is not None:
        atr = compute_atr(highs, lows, closes, atr_period)
    else:
        atr = compute_atr_from_closes(closes, atr_period)

    atr_pct = atr / entry_price if entry_price > 0 else 0.02

    # Apply regime multipliers (or overrides)
    sl_mult = sl_atr_mult if sl_atr_mult is not None else _REGIME_SL_MULT.get(regime, 1.5)
    tp_mult = tp_atr_mult if tp_atr_mult is not None else _REGIME_TP_MULT.get(regime, 2.0)

    # Compute raw SL/TP as % of price
    sl_pct = atr_pct * sl_mult
    tp_pct = atr_pct * tp_mult

    # Clamp to floor/ceiling
    sl_pct = max(_MIN_SL_PCT, min(_MAX_SL_PCT, sl_pct))
    tp_pct = max(_MIN_TP_PCT, min(_MAX_TP_PCT, tp_pct))

    # Compute actual prices
    if direction == "LONG":
        stop_loss_price = round(entry_price * (1 - sl_pct), 2)
        take_profit_price = round(entry_price * (1 + tp_pct), 2)
    else:  # SHORT
        stop_loss_price = round(entry_price * (1 + sl_pct), 2)
        take_profit_price = round(entry_price * (1 - tp_pct), 2)

    logger.info(
        "ATR brackets %s %s: ATR=$%.2f (%.1f%%), SL=%.1f%% ($%.2f), TP=%.1f%% ($%.2f) [%s]",
        direction, ticker, atr, atr_pct * 100,
        sl_pct * 100, stop_loss_price,
        tp_pct * 100, take_profit_price,
        regime,
    )

    return BracketLevels(
        ticker=ticker,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        stop_loss_pct=round(sl_pct, 6),
        take_profit_pct=round(tp_pct, 6),
        atr_value=round(atr, 4),
        atr_pct=round(atr_pct, 6),
        regime=regime,
        direction=direction,
    )


def calculate_brackets_multi(
    price_df: pd.DataFrame,
    tickers: list[str],
    regime: str = "NORMAL",
) -> Dict[str, BracketLevels]:
    """Compute bracket levels for multiple tickers at once.

    Parameters
    ----------
    price_df : pd.DataFrame
        Columns are ticker symbols, values are close prices.
    tickers : list[str]
        Tickers to compute brackets for.
    regime : str
        Current market regime.

    Returns
    -------
    dict
        Mapping of ticker → BracketLevels.
    """
    results = {}
    for ticker in tickers:
        if ticker in price_df.columns:
            closes = price_df[ticker].dropna()
            if len(closes) > 0:
                entry = float(closes.iloc[-1])
                results[ticker] = calculate_brackets(
                    ticker=ticker,
                    entry_price=entry,
                    closes=closes,
                    regime=regime,
                )
    return results
