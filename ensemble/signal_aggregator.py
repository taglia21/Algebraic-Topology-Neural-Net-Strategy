"""
ensemble/signal_aggregator.py
=============================
Combine TDA and NN signals using meta-allocator weights into a single
ranked list of actionable signals.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bonus / penalty multipliers
_AGREEMENT_BONUS = 1.20   # +20% when strategies agree
_DISAGREEMENT_PENALTY = 0.70  # -30% when strategies strongly disagree


class SignalAggregator:
    """Aggregate TDA and NN signals into final trading decisions.

    Parameters
    ----------
    agreement_bonus : float
        Multiplier applied to strength when both strategies agree on
        direction (default 1.20 → +20%).
    disagreement_penalty : float
        Multiplier applied when strategies strongly disagree (default 0.70
        → -30%).
    """

    def __init__(
        self,
        agreement_bonus: float = _AGREEMENT_BONUS,
        disagreement_penalty: float = _DISAGREEMENT_PENALTY,
    ) -> None:
        self.agreement_bonus = agreement_bonus
        self.disagreement_penalty = disagreement_penalty
        self._last_aggregated: Optional[pd.DataFrame] = None

    def aggregate(
        self,
        tda_signals: pd.DataFrame,
        nn_signals: pd.DataFrame,
        tda_weight: float,
        nn_weight: float,
    ) -> pd.DataFrame:
        """Combine TDA and NN signals using allocation weights.

        Parameters
        ----------
        tda_signals : pd.DataFrame
            Must have columns: ticker, direction, strength.
        nn_signals : pd.DataFrame
            Must have columns: ticker, direction, strength.
        tda_weight : float
            Weight for TDA strategy (0–1).
        nn_weight : float
            Weight for NN strategy (0–1).

        Returns
        -------
        pd.DataFrame
            Columns: ticker, direction, final_strength, tda_component,
            nn_component, agreement, timestamp.
        """
        # Build lookup dicts keyed by ticker
        tda_map = {}
        if not tda_signals.empty:
            for _, row in tda_signals.iterrows():
                tda_map[row["ticker"]] = {
                    "direction": row["direction"],
                    "strength": float(row["strength"]),
                    "timestamp": row.get("timestamp"),
                }

        nn_map = {}
        if not nn_signals.empty:
            for _, row in nn_signals.iterrows():
                nn_map[row["ticker"]] = {
                    "direction": row["direction"],
                    "strength": float(row["strength"]),
                    "timestamp": row.get("timestamp"),
                }

        all_tickers = set(tda_map.keys()) | set(nn_map.keys())
        records = []

        for ticker in sorted(all_tickers):
            tda_info = tda_map.get(ticker, {"direction": "NEUTRAL", "strength": 0.0, "timestamp": None})
            nn_info = nn_map.get(ticker, {"direction": "NEUTRAL", "strength": 0.0, "timestamp": None})

            tda_dir = tda_info["direction"]
            nn_dir = nn_info["direction"]
            tda_str = tda_info["strength"]
            nn_str = nn_info["strength"]
            timestamp = tda_info["timestamp"] or nn_info["timestamp"]

            # Weighted components
            tda_component = tda_weight * tda_str
            nn_component = nn_weight * nn_str

            # Check agreement
            both_active = tda_dir != "NEUTRAL" and nn_dir != "NEUTRAL"
            agreement = both_active and tda_dir == nn_dir
            strong_disagreement = (
                both_active
                and tda_dir != nn_dir
            )

            # Resolve direction
            if tda_dir == nn_dir:
                direction = tda_dir
            elif tda_dir == "NEUTRAL":
                direction = nn_dir
            elif nn_dir == "NEUTRAL":
                direction = tda_dir
            else:
                # Strategies disagree — go with the stronger weighted signal
                if tda_component > nn_component:
                    direction = tda_dir
                elif nn_component > tda_component:
                    direction = nn_dir
                else:
                    # Equal strength and disagreeing → NEUTRAL
                    direction = "NEUTRAL"

            # Compute final strength
            raw_strength = tda_component + nn_component

            if agreement:
                raw_strength *= self.agreement_bonus
            elif strong_disagreement:
                raw_strength *= self.disagreement_penalty

            final_strength = min(1.0, max(0.0, raw_strength))

            # If direction resolved to NEUTRAL, zero out strength
            if direction == "NEUTRAL":
                final_strength = 0.0

            records.append({
                "ticker": ticker,
                "direction": direction,
                "final_strength": round(final_strength, 6),
                "tda_component": round(tda_component, 6),
                "nn_component": round(nn_component, 6),
                "agreement": agreement,
                "timestamp": timestamp,
            })

        result = pd.DataFrame(records)
        if result.empty:
            result = pd.DataFrame(
                columns=[
                    "ticker", "direction", "final_strength",
                    "tda_component", "nn_component", "agreement", "timestamp",
                ]
            )
        self._last_aggregated = result
        return result

    def filter_signals(self, min_strength: float = 0.3) -> pd.DataFrame:
        """Return only signals above minimum strength threshold.

        Parameters
        ----------
        min_strength : float
            Minimum final_strength to include (default 0.3).

        Returns
        -------
        pd.DataFrame
            Filtered signals.
        """
        if self._last_aggregated is None or self._last_aggregated.empty:
            return pd.DataFrame(
                columns=[
                    "ticker", "direction", "final_strength",
                    "tda_component", "nn_component", "agreement", "timestamp",
                ]
            )

        mask = (
            (self._last_aggregated["final_strength"] >= min_strength)
            & (self._last_aggregated["direction"] != "NEUTRAL")
        )
        return self._last_aggregated[mask].reset_index(drop=True)

    def rank_signals(self) -> pd.DataFrame:
        """Return all non-neutral signals sorted by strength descending.

        Returns
        -------
        pd.DataFrame
            Ranked signals.
        """
        if self._last_aggregated is None or self._last_aggregated.empty:
            return pd.DataFrame(
                columns=[
                    "ticker", "direction", "final_strength",
                    "tda_component", "nn_component", "agreement", "timestamp",
                ]
            )

        active = self._last_aggregated[
            self._last_aggregated["direction"] != "NEUTRAL"
        ]
        return (
            active.sort_values("final_strength", ascending=False)
            .reset_index(drop=True)
        )
