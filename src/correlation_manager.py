"""
Cross-Asset Correlation Monitor
================================

Institutional-grade correlation monitoring with:
  - 21d (short) and 63d (medium) rolling correlation matrices
  - Cross-asset stress indicators: VIX, TLT, HYG, GLD
  - Correlation breakdown detection (regime-driven decorrelation)
  - Aggregate correlation risk score for portfolio gating

Wired into risk_guardian.validate_entry() to block trades when
correlation risk is elevated (everything moving together → systemic risk).

Author: Phase 4 — Institutional-Grade Integration
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Stress-correlated macro assets
CROSS_ASSETS = ["VIX", "TLT", "HYG", "GLD"]


@dataclass
class CorrelationReport:
    """Result of cross-asset correlation analysis."""
    avg_corr_21d: float            # mean absolute pairwise corr (21d)
    avg_corr_63d: float            # mean absolute pairwise corr (63d)
    max_corr_pair: Tuple[str, str] # highest pairwise correlation pair
    max_corr_value: float          # value of that pair
    cross_asset_scores: Dict[str, float]  # {VIX: corr_w_portfolio, TLT: ..., ...}
    breakdown_detected: bool       # True if correlations spiking
    risk_score: float              # 0-100 aggregate risk
    timestamp: datetime = field(default_factory=datetime.now)

    def describe(self) -> str:
        pair = f"{self.max_corr_pair[0]}/{self.max_corr_pair[1]}"
        return (
            f"corr_risk={self.risk_score:.0f}/100  "
            f"avg21d={self.avg_corr_21d:.2f}  avg63d={self.avg_corr_63d:.2f}  "
            f"max_pair={pair}={self.max_corr_value:.2f}  "
            f"breakdown={'YES' if self.breakdown_detected else 'no'}"
        )


class CrossAssetCorrelationMonitor:
    """
    Monitor rolling pairwise correlations & cross-asset stress signals.

    Parameters
    ----------
    short_window : int
        Short rolling window (default 21 trading days ≈ 1 month).
    medium_window : int
        Medium rolling window (default 63 trading days ≈ 3 months).
    breakdown_threshold : float
        If avg_corr_21d exceeds this, a correlation breakdown is flagged
        (everything correlating → systemic risk). Default 0.70.
    risk_score_threshold : float
        Risk score above which validate_entry should block new trades.
        Default 70.
    """

    def __init__(
        self,
        short_window: int = 21,
        medium_window: int = 63,
        breakdown_threshold: float = 0.70,
        risk_score_threshold: float = 70.0,
    ):
        self.short_window = short_window
        self.medium_window = medium_window
        self.breakdown_threshold = breakdown_threshold
        self.risk_score_threshold = risk_score_threshold

        # Internal state
        self._returns_buffer: Dict[str, List[float]] = {}
        self._last_report: Optional[CorrelationReport] = None
        self._report_cache_ts: Optional[datetime] = None
        self._cache_ttl_sec: int = 300  # 5-min cache
        self.logger = logging.getLogger(f"{__name__}.CrossAssetCorrelation")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """
        Feed a single-day return for a symbol.

        Call once per day per symbol held (plus cross-assets).
        Buffer is kept to medium_window + 10 to allow rolling.
        """
        buf = self._returns_buffer.setdefault(symbol, [])
        buf.append(daily_return)
        max_len = self.medium_window + 10
        if len(buf) > max_len:
            self._returns_buffer[symbol] = buf[-max_len:]

    def load_returns_matrix(
        self, returns_dict: Dict[str, np.ndarray]
    ) -> None:
        """
        Bulk-load historical returns (e.g. from price_data).

        Parameters
        ----------
        returns_dict : dict
            {symbol: np.ndarray of daily log returns}
        """
        for symbol, rets in returns_dict.items():
            self._returns_buffer[symbol] = list(rets[-(self.medium_window + 10):])

    def analyze(self, portfolio_symbols: List[str]) -> CorrelationReport:
        """
        Compute correlation report for the given portfolio.

        Parameters
        ----------
        portfolio_symbols : list[str]
            Currently held symbols (excludes cross-assets, which are
            tracked automatically if present in the buffer).

        Returns
        -------
        CorrelationReport
        """
        # Cache check
        now = datetime.now()
        if (
            self._last_report is not None
            and self._report_cache_ts is not None
            and (now - self._report_cache_ts).total_seconds() < self._cache_ttl_sec
        ):
            return self._last_report

        # Gather aligned return matrix for portfolio
        avg_21, avg_63, max_pair, max_val = self._compute_rolling_corrs(
            portfolio_symbols
        )

        # Cross-asset stress signals
        cross_scores = self._compute_cross_asset_corr(portfolio_symbols)

        # Breakdown detection
        breakdown = avg_21 > self.breakdown_threshold

        # Aggregate risk score
        risk = self._compute_risk_score(avg_21, avg_63, cross_scores, breakdown)

        report = CorrelationReport(
            avg_corr_21d=avg_21,
            avg_corr_63d=avg_63,
            max_corr_pair=max_pair,
            max_corr_value=max_val,
            cross_asset_scores=cross_scores,
            breakdown_detected=breakdown,
            risk_score=risk,
        )
        self._last_report = report
        self._report_cache_ts = now

        if breakdown:
            self.logger.warning(f"Correlation breakdown!  {report.describe()}")
        else:
            self.logger.debug(f"Correlation report: {report.describe()}")

        return report

    def detect_correlation_breakdown(self, portfolio_symbols: List[str]) -> bool:
        """Quick check: is average 21d correlation above breakdown threshold?"""
        report = self.analyze(portfolio_symbols)
        return report.breakdown_detected

    def correlation_risk_score(self, portfolio_symbols: List[str]) -> float:
        """Return 0-100 aggregate risk score."""
        report = self.analyze(portfolio_symbols)
        return report.risk_score

    def should_block_entry(self, portfolio_symbols: List[str]) -> Tuple[bool, str]:
        """
        Whether correlation risk is too high to allow new entries.

        Returns (block, reason).
        """
        report = self.analyze(portfolio_symbols)
        if report.risk_score >= self.risk_score_threshold:
            return True, (
                f"Correlation risk score {report.risk_score:.0f}/100 "
                f">= {self.risk_score_threshold:.0f} threshold  "
                f"(avg21d={report.avg_corr_21d:.2f})"
            )
        return False, "ok"

    # ------------------------------------------------------------------ #
    # Internal computation
    # ------------------------------------------------------------------ #

    def _get_aligned_matrix(
        self, symbols: List[str], window: int
    ) -> Optional[np.ndarray]:
        """Build (window, n_symbols) aligned return matrix. Returns None if <2 symbols."""
        valid = []
        arrays = []
        for s in symbols:
            buf = self._returns_buffer.get(s)
            if buf is not None and len(buf) >= window:
                arrays.append(np.array(buf[-window:]))
                valid.append(s)
        if len(valid) < 2:
            return None
        # Stack as columns (each column = one symbol's returns)
        return np.column_stack(arrays)

    def _compute_rolling_corrs(
        self, symbols: List[str]
    ) -> Tuple[float, float, Tuple[str, str], float]:
        """
        Compute average absolute pairwise correlations for 21d and 63d windows.

        Returns (avg_21d, avg_63d, max_corr_pair, max_corr_value).
        """
        avg_21 = 0.0
        avg_63 = 0.0
        max_pair: Tuple[str, str] = ("", "")
        max_val = 0.0

        valid_syms = [s for s in symbols if s in self._returns_buffer]

        # 21-day
        mat_21 = self._get_aligned_matrix(valid_syms, self.short_window)
        if mat_21 is not None and mat_21.shape[1] >= 2:
            corr_21 = np.corrcoef(mat_21.T)
            n = corr_21.shape[0]
            upper = []
            for i in range(n):
                for j in range(i + 1, n):
                    c = abs(float(corr_21[i, j]))
                    upper.append(c)
                    if c > max_val:
                        max_val = c
                        max_pair = (valid_syms[i], valid_syms[j])
            avg_21 = float(np.mean(upper)) if upper else 0.0

        # 63-day
        mat_63 = self._get_aligned_matrix(valid_syms, self.medium_window)
        if mat_63 is not None and mat_63.shape[1] >= 2:
            corr_63 = np.corrcoef(mat_63.T)
            n = corr_63.shape[0]
            upper = []
            for i in range(n):
                for j in range(i + 1, n):
                    upper.append(abs(float(corr_63[i, j])))
            avg_63 = float(np.mean(upper)) if upper else 0.0

        return avg_21, avg_63, max_pair, max_val

    def _compute_cross_asset_corr(
        self, portfolio_symbols: List[str]
    ) -> Dict[str, float]:
        """
        Compute correlation of portfolio equal-weighted return stream with
        each stress-indicator asset (VIX, TLT, HYG, GLD).
        """
        scores: Dict[str, float] = {}
        # Build portfolio return series (equal-weighted)
        valid_port = [
            s for s in portfolio_symbols
            if s in self._returns_buffer and len(self._returns_buffer[s]) >= self.short_window
        ]
        if not valid_port:
            return {ca: 0.0 for ca in CROSS_ASSETS}

        port_mat = self._get_aligned_matrix(valid_port, self.short_window)
        if port_mat is None:
            return {ca: 0.0 for ca in CROSS_ASSETS}
        port_ret = port_mat.mean(axis=1)  # equal-weight daily return

        for ca in CROSS_ASSETS:
            buf = self._returns_buffer.get(ca)
            if buf is None or len(buf) < self.short_window:
                scores[ca] = 0.0
                continue
            ca_ret = np.array(buf[-self.short_window:])
            if len(ca_ret) != len(port_ret):
                scores[ca] = 0.0
                continue
            corr_val = float(np.corrcoef(port_ret, ca_ret)[0, 1])
            scores[ca] = corr_val if not np.isnan(corr_val) else 0.0

        return scores

    def _compute_risk_score(
        self,
        avg_21: float,
        avg_63: float,
        cross_scores: Dict[str, float],
        breakdown: bool,
    ) -> float:
        """
        Aggregate 0-100 risk score.

        Components (weighted):
          40% — avg 21d absolute correlation (0→0, 1→100)
          20% — avg 63d absolute correlation
          15% — |corr with VIX| (positive corr with VIX = bad)
          10% — |corr with HYG| (negative corr with HYG = flight from risk)
          15% — breakdown flag (adds flat 15 points if triggered)
        """
        s = 0.0
        s += 40.0 * min(avg_21, 1.0)
        s += 20.0 * min(avg_63, 1.0)

        vix_corr = abs(cross_scores.get("VIX", 0.0))
        hyg_corr = abs(cross_scores.get("HYG", 0.0))
        s += 15.0 * min(vix_corr, 1.0)
        s += 10.0 * min(hyg_corr, 1.0)
        if breakdown:
            s += 15.0

        return min(round(s, 1), 100.0)
