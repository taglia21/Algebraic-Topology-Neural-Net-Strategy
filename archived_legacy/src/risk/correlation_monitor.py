"""
Phase R — Correlation Breakdown Monitor.

Item 18: CorrelationBreakdownMonitor — 20d vs 252d correlation, sign flip detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAlert:
    """Alert for correlation breakdown."""
    asset_i: str
    asset_j: str
    short_term_corr: float
    long_term_corr: float
    change: float
    is_sign_flip: bool
    severity: str  # "info", "warning", "critical"
    timestamp: str = ""


@dataclass
class CorrelationReport:
    """Correlation monitoring report."""
    short_term_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    long_term_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    change_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    alerts: List[CorrelationAlert] = field(default_factory=list)
    sign_flips: int = 0
    max_change: float = 0.0
    avg_change: float = 0.0
    n_assets: int = 0
    is_breakdown: bool = False  # True if any critical alerts


class CorrelationBreakdownMonitor:
    """Monitor correlation regime changes.

    Compares short-term (20d) vs long-term (252d) correlation matrices.
    Detects:
      - Large absolute changes in correlation.
      - Sign flips (positive → negative or vice versa).
      - Correlation convergence to 1 (crisis indicator).

    Alert levels:
      - info: |change| > 0.3
      - warning: |change| > 0.5 or sign flip
      - critical: sign flip + |change| > 0.5
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 252,
        change_threshold: float = 0.3,
        critical_threshold: float = 0.5,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.change_threshold = change_threshold
        self.critical_threshold = critical_threshold
        self._history: List[CorrelationReport] = []

    def compute_correlation(
        self,
        returns: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute correlation matrix from trailing returns.

        Args:
            returns: (T, N) return matrix.
            window: Lookback window.

        Returns:
            (N, N) correlation matrix.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.shape[0] < window:
            window = returns.shape[0]
        if window < 3:
            n = returns.shape[1]
            return np.eye(n)

        recent = returns[-window:]
        corr = np.corrcoef(recent, rowvar=False)
        # Handle NaN
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def monitor(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> CorrelationReport:
        """Run correlation breakdown analysis.

        Args:
            returns: (T, N) matrix of daily returns.
            asset_names: Names for the N assets.

        Returns:
            CorrelationReport with alerts.
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = returns.shape[1]

        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n)]

        short_corr = self.compute_correlation(returns, self.short_window)
        long_corr = self.compute_correlation(returns, self.long_window)
        change = short_corr - long_corr

        alerts = []
        sign_flips = 0

        for i in range(n):
            for j in range(i + 1, n):
                abs_change = abs(change[i, j])
                is_flip = (short_corr[i, j] * long_corr[i, j] < 0) and abs(long_corr[i, j]) > 0.1

                if is_flip:
                    sign_flips += 1

                if abs_change < self.change_threshold and not is_flip:
                    continue

                # Determine severity
                if is_flip and abs_change > self.critical_threshold:
                    severity = "critical"
                elif is_flip or abs_change > self.critical_threshold:
                    severity = "warning"
                else:
                    severity = "info"

                alerts.append(CorrelationAlert(
                    asset_i=asset_names[i],
                    asset_j=asset_names[j],
                    short_term_corr=float(short_corr[i, j]),
                    long_term_corr=float(long_corr[i, j]),
                    change=float(change[i, j]),
                    is_sign_flip=is_flip,
                    severity=severity,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

        # Extract upper triangle for stats
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        abs_changes = np.abs(change[mask])
        max_change = float(np.max(abs_changes)) if len(abs_changes) > 0 else 0.0
        avg_change = float(np.mean(abs_changes)) if len(abs_changes) > 0 else 0.0

        is_breakdown = any(a.severity == "critical" for a in alerts)

        report = CorrelationReport(
            short_term_matrix=short_corr,
            long_term_matrix=long_corr,
            change_matrix=change,
            alerts=alerts,
            sign_flips=sign_flips,
            max_change=max_change,
            avg_change=avg_change,
            n_assets=n,
            is_breakdown=is_breakdown,
        )

        self._history.append(report)

        if is_breakdown:
            logger.warning(
                "CORRELATION BREAKDOWN: %d sign flips, max change=%.3f",
                sign_flips, max_change,
            )
        else:
            logger.info(
                "Correlation stable: %d alerts, avg change=%.3f",
                len(alerts), avg_change,
            )

        return report

    @property
    def history(self) -> List[CorrelationReport]:
        return self._history
