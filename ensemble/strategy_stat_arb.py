"""
ensemble/strategy_stat_arb.py
=============================
Cross-Sectional Statistical Arbitrage Strategy (ORIA alpha sleeve #3).

Detects relative-value mispricings within sector pairs and across
correlated assets, generating pair-neutral signals.

This is the most independent alpha source — it operates on spread
dynamics between related assets rather than absolute levels.

Features:
- Sector-pair spread z-scores
- Cross-sectional rank deviation (stocks that moved too far from their peers)
- Rolling correlation breakdowns as trade triggers
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REGIME_DISCOUNT = {"NORMAL": 1.0, "STRESSED": 0.80, "CRASH": 0.50}

# Predefined pairs with strong fundamental linkage (expanded for 50-symbol universe)
_PAIRS = [
    # Big tech peers
    ("AAPL", "MSFT"),
    ("AMZN", "GOOGL"),
    ("NVDA", "AMD"),     # GPU rivals
    ("META", "NFLX"),    # Digital media
    ("AVGO", "INTC"),    # Semiconductor
    ("CRM", "ORCL"),     # Enterprise software
    ("ADBE", "CRM"),     # SaaS
    # Finance pairs
    ("JPM", "BAC"),      # Big banks
    ("GS", "JPM"),       # Investment banks
    ("V", "MA"),          # Payments
    ("XLF", "JPM"),      # Sector vs leader
    # Healthcare pairs
    ("JNJ", "PFE"),      # Pharma
    ("UNH", "ABBV"),     # Healthcare giants
    ("MRK", "PFE"),      # Pharma competitors
    # Consumer pairs
    ("WMT", "COST"),     # Retail
    ("HD", "WMT"),       # Consumer spend
    ("MCD", "SBUX"),     # Quick service
    ("NKE", "DIS"),      # Consumer brands
    # Energy pairs
    ("XOM", "CVX"),      # Oil majors
    ("COP", "XOM"),      # E&P vs integrated
    ("XLE", "XOM"),      # Sector vs leader
    # Industrials
    ("CAT", "BA"),       # Industrial giants
    ("UPS", "BA"),       # Transport/industrial
    # Cross-asset
    ("GLD", "TLT"),      # Safe havens
    ("SPY", "QQQ"),      # Broad vs tech
    ("XLE", "XLI"),      # Cyclical sectors
    ("IWM", "SPY"),      # Small vs large cap
]


class StatArbStrategy:
    """Cross-sectional statistical arbitrage alpha sleeve.

    Parameters
    ----------
    spread_window : int
        Window for computing spread mean and std (default 30).
    z_threshold : float
        Z-score threshold for signal generation (default 1.5).
    rank_deviation_threshold : float
        Cross-sectional rank deviation threshold (default 0.3).
    """

    def __init__(
        self,
        spread_window: int = 30,
        z_threshold: float = 1.5,
        rank_deviation_threshold: float = 0.3,
    ):
        self.spread_window = spread_window
        self.z_threshold = z_threshold
        self.rank_deviation_threshold = rank_deviation_threshold

    def _compute_spread_zscore(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> Optional[float]:
        """Compute z-score of the log-price spread between two assets."""
        if len(prices_a) < self.spread_window + 5 or len(prices_b) < self.spread_window + 5:
            return None

        # Log-price ratio spread
        spread = np.log(prices_a / prices_b)
        spread = spread.dropna()
        if len(spread) < self.spread_window:
            return None

        recent = spread.tail(self.spread_window)
        mean = recent.mean()
        std = recent.std()

        if std < 1e-8:
            return None

        current = spread.iloc[-1]
        return (current - mean) / std

    def _compute_rank_deviation(
        self,
        returns_df: pd.DataFrame,
        window: int = 10,
    ) -> Dict[str, float]:
        """Compute how much each stock's recent rank deviates from its typical rank.

        Stocks that suddenly jump or drop in the cross-sectional ranking
        may revert to their typical position.
        """
        if returns_df is None or returns_df.empty or len(returns_df) < window + 10:
            return {}

        # Compute cumulative returns over last `window` days
        recent_cum = (1 + returns_df.tail(window)).cumprod().iloc[-1] - 1
        prior_cum = (1 + returns_df.iloc[-(window * 2):-window]).cumprod().iloc[-1] - 1

        # Rank both periods
        recent_ranks = recent_cum.rank(pct=True)
        prior_ranks = prior_cum.rank(pct=True)

        deviations = {}
        for ticker in recent_ranks.index:
            if ticker in prior_ranks.index:
                dev = recent_ranks[ticker] - prior_ranks[ticker]
                deviations[ticker] = float(dev)

        return deviations

    def generate_signals(
        self,
        price_df: pd.DataFrame,
        returns_df: Optional[pd.DataFrame] = None,
        regime: str = "NORMAL",
    ) -> pd.DataFrame:
        """Generate stat-arb signals from price and return data.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price data.
        returns_df : pd.DataFrame, optional
            Return data (will be computed from price_df if None).
        regime : str
            Current regime.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, direction, strength, regime, timestamp.
        """
        if price_df is None or price_df.empty or len(price_df) < self.spread_window + 10:
            return pd.DataFrame(columns=["ticker", "direction", "strength", "regime", "timestamp"])

        if returns_df is None:
            returns_df = price_df.pct_change().dropna()

        records = []
        discount = _REGIME_DISCOUNT.get(regime, 1.0)
        timestamp = price_df.index[-1] if hasattr(price_df.index, '__len__') else None
        seen_tickers = set()

        # === 1. Pair spread signals ===
        for a, b in _PAIRS:
            if a not in price_df.columns or b not in price_df.columns:
                continue

            z = self._compute_spread_zscore(price_df[a], price_df[b])
            if z is None:
                continue

            if abs(z) >= self.z_threshold:
                # Spread is extreme → expect mean reversion
                raw_strength = min(1.0, (abs(z) - self.z_threshold) / self.z_threshold)

                if z > self.z_threshold:
                    # A overpriced relative to B → short A, long B
                    if a not in seen_tickers:
                        records.append({
                            "ticker": a, "direction": "SHORT",
                            "strength": round(raw_strength * discount * 0.7, 6),
                            "regime": regime, "timestamp": timestamp,
                        })
                        seen_tickers.add(a)
                    if b not in seen_tickers:
                        records.append({
                            "ticker": b, "direction": "LONG",
                            "strength": round(raw_strength * discount * 0.7, 6),
                            "regime": regime, "timestamp": timestamp,
                        })
                        seen_tickers.add(b)
                else:
                    # A underpriced relative to B → long A, short B
                    if a not in seen_tickers:
                        records.append({
                            "ticker": a, "direction": "LONG",
                            "strength": round(raw_strength * discount * 0.7, 6),
                            "regime": regime, "timestamp": timestamp,
                        })
                        seen_tickers.add(a)
                    if b not in seen_tickers:
                        records.append({
                            "ticker": b, "direction": "SHORT",
                            "strength": round(raw_strength * discount * 0.7, 6),
                            "regime": regime, "timestamp": timestamp,
                        })
                        seen_tickers.add(b)

        # === 2. Cross-sectional rank deviation signals ===
        rank_devs = self._compute_rank_deviation(returns_df)
        for ticker, dev in rank_devs.items():
            if ticker in seen_tickers:
                continue
            if abs(dev) >= self.rank_deviation_threshold:
                raw_strength = min(0.6, abs(dev))
                if dev > 0:
                    # Recently outperformed peers → expect reversion → SHORT
                    records.append({
                        "ticker": ticker, "direction": "SHORT",
                        "strength": round(raw_strength * discount, 6),
                        "regime": regime, "timestamp": timestamp,
                    })
                else:
                    # Recently underperformed peers → expect reversion → LONG
                    records.append({
                        "ticker": ticker, "direction": "LONG",
                        "strength": round(raw_strength * discount, 6),
                        "regime": regime, "timestamp": timestamp,
                    })
                seen_tickers.add(ticker)

        result = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["ticker", "direction", "strength", "regime", "timestamp"]
        )

        logger.info(
            "StatArbStrategy: %d signals (%d pair, %d rank_dev) | regime=%s",
            len(result),
            sum(1 for r in records if r.get("strength", 0) < 0.5),
            sum(1 for r in records if r.get("strength", 0) >= 0.5) if records else 0,
            regime,
        )
        return result
