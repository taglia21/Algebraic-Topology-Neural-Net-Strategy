"""
core/paper_returns.py
=====================
Collect hypothetical (paper) returns for both TDA and NN strategies
independently, to eventually train the MetaAllocator.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/paper_returns.jsonl")


class PaperReturnsCollector:
    """Collect daily paper returns for strategy comparison."""

    def __init__(self, path: Path = _DEFAULT_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        date: str,
        tda_signals: pd.DataFrame,
        nn_signals: pd.DataFrame,
        returns_df: pd.DataFrame,
        regime: str = "NORMAL",
    ) -> None:
        """Record hypothetical strategy returns for one trading day.

        For each strategy, compute what the return would have been
        if we followed all signals equally weighted.
        """
        tda_return = self._compute_strategy_return(tda_signals, returns_df)
        nn_return = self._compute_strategy_return(nn_signals, returns_df)

        record = {
            "date": date,
            "tda_return": tda_return,
            "nn_return": nn_return,
            "regime": regime,
            "tda_signal_count": len(tda_signals) if not tda_signals.empty else 0,
            "nn_signal_count": len(nn_signals) if not nn_signals.empty else 0,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug("Recorded paper returns: TDA=%.4f, NN=%.4f", tda_return, nn_return)
        except Exception as e:
            logger.warning("Failed to record paper returns: %s", e)

    def _compute_strategy_return(self, signals: pd.DataFrame,
                                  returns_df: pd.DataFrame) -> float:
        """Compute equal-weight return from signals using next-day returns."""
        if signals.empty or returns_df.empty:
            return 0.0

        total_return = 0.0
        count = 0

        for _, sig in signals.iterrows():
            ticker = sig.get("ticker", "")
            direction = sig.get("direction", "NEUTRAL")
            strength = float(sig.get("strength", 0.0))

            if direction == "NEUTRAL" or strength < 0.3:
                continue

            if ticker in returns_df.columns and len(returns_df) > 0:
                next_ret = returns_df[ticker].iloc[-1]  # Latest return
                if direction == "LONG":
                    total_return += next_ret * strength
                elif direction == "SHORT":
                    total_return -= next_ret * strength
                count += 1

        return total_return / max(count, 1)

    def load_history(self) -> pd.DataFrame:
        """Load all recorded paper returns."""
        if not self._path.exists():
            return pd.DataFrame()

        records = []
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.warning("Failed to load paper returns: %s", e)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_summary(self) -> Dict:
        """Get summary statistics of paper returns."""
        df = self.load_history()
        if df.empty:
            return {"days": 0, "tda_total": 0.0, "nn_total": 0.0}

        return {
            "days": len(df),
            "tda_total": float(df["tda_return"].sum()),
            "nn_total": float(df["nn_return"].sum()),
            "tda_mean": float(df["tda_return"].mean()),
            "nn_mean": float(df["nn_return"].mean()),
            "tda_sharpe": float(df["tda_return"].mean() / df["tda_return"].std() * (252 ** 0.5)) if df["tda_return"].std() > 0 else 0.0,
            "nn_sharpe": float(df["nn_return"].mean() / df["nn_return"].std() * (252 ** 0.5)) if df["nn_return"].std() > 0 else 0.0,
        }
