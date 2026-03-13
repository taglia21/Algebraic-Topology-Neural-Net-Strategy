"""
ensemble/risk_manager.py
========================
Regime-aware position sizing and risk management for the ensemble module.

Computes position sizes using fractional Kelly criterion, enforces regime-based
exposure caps, and integrates with broker kill-switch thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Computed position size for a single signal."""

    ticker: str
    direction: str
    raw_kelly_fraction: float
    position_pct: float  # % of NAV
    position_value: float  # $ amount
    regime: str
    capped: bool = False  # True if position was capped by a limit


@dataclass
class RiskReport:
    """Portfolio-level risk assessment."""

    total_long_exposure: float
    total_short_exposure: float
    gross_exposure: float
    net_exposure: float
    max_single_position_pct: float
    sector_exposures: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)


# Regime-based exposure limits
# Tuned for small-to-mid accounts ($5,444).
_REGIME_LIMITS = {
    "NORMAL": {
        "max_position_pct": 20.0,
        "max_total_exposure_pct": 100.0,
    },
    "STRESSED": {
        "max_position_pct": 12.0,
        "max_total_exposure_pct": 60.0,
    },
    "CRASH": {
        "max_position_pct": 8.0,
        "max_total_exposure_pct": 30.0,
    },
}


class EnsembleRiskManager:
    """Position sizing and risk management for ensemble signals.

    Parameters
    ----------
    max_position_pct : float
        Maximum single position as % of NAV (default 5.0).
    max_sector_exposure_pct : float
        Maximum sector exposure as % of NAV (default 20.0).
    max_long_exposure_pct : float
        Maximum total long exposure as % of NAV (default 100.0).
    max_short_exposure_pct : float
        Maximum total short exposure as % of NAV (default 50.0).
    max_gross_exposure_pct : float
        Maximum gross exposure as % of NAV (default 130.0).
    kelly_multiplier : float
        Fraction of full Kelly to use (default 0.5 = half-Kelly).
    daily_loss_flatten_pct : float
        Daily loss % that triggers full flatten (default 5.0).
    daily_loss_reduce_pct : float
        Daily loss % that triggers 50% exposure reduction (default 3.0).
    max_drawdown_halt_pct : float
        Maximum drawdown % that halts the system (default 15.0).
    max_risk_per_trade : float
        Max dollar risk per trade for small accounts (default 50.0).
    max_option_premium : float
        Max option premium outlay for small accounts (default 50.0).
    max_equity_position : float
        Max equity position size for small accounts (default 100.0).
    """

    def __init__(
        self,
        max_position_pct: float = 5.0,
        max_sector_exposure_pct: float = 20.0,
        max_long_exposure_pct: float = 100.0,
        max_short_exposure_pct: float = 50.0,
        max_gross_exposure_pct: float = 130.0,
        kelly_multiplier: float = 0.5,
        daily_loss_flatten_pct: float = 5.0,
        daily_loss_reduce_pct: float = 3.0,
        max_drawdown_halt_pct: float = 15.0,
        max_risk_per_trade: float = 500.0,
        max_option_premium: float = 50.0,
        max_equity_position: float = 1000.0,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_long_exposure_pct = max_long_exposure_pct
        self.max_short_exposure_pct = max_short_exposure_pct
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.kelly_multiplier = kelly_multiplier
        self.daily_loss_flatten_pct = daily_loss_flatten_pct
        self.daily_loss_reduce_pct = daily_loss_reduce_pct
        self.max_drawdown_halt_pct = max_drawdown_halt_pct
        self.max_risk_per_trade = max_risk_per_trade
        self.max_option_premium = max_option_premium
        self.max_equity_position = max_equity_position

    @staticmethod
    def kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute the Kelly criterion fraction.

        Parameters
        ----------
        win_rate : float
            Probability of winning (0–1).
        avg_win : float
            Average winning trade return (positive).
        avg_loss : float
            Average losing trade return (positive, magnitude).

        Returns
        -------
        float
            Optimal Kelly fraction. Returns 0 if edge is negative.
        """
        if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        b = avg_win / avg_loss  # odds ratio
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b
        return max(0.0, kelly)

    def size_position(
        self,
        signal: Dict,
        portfolio_value: float,
        current_exposure: Dict[str, float],
        regime: str = "NORMAL",
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015,
    ) -> PositionSize:
        """Compute position size for a single signal.

        Parameters
        ----------
        signal : dict
            Must contain ``'ticker'``, ``'direction'``, ``'strength'``.
        portfolio_value : float
            Current portfolio NAV in dollars.
        current_exposure : dict
            Keys: ``'long_pct'``, ``'short_pct'``, ``'gross_pct'``.
        regime : str
            Current market regime.
        win_rate : float
            Historical win rate for Kelly calculation (default 0.55).
        avg_win : float
            Average win magnitude (default 0.02).
        avg_loss : float
            Average loss magnitude (default 0.015).

        Returns
        -------
        PositionSize
            Computed position sizing.
        """
        ticker = signal["ticker"]
        direction = signal["direction"]
        strength = float(signal.get("strength", 0.0))

        if direction == "NEUTRAL" or strength <= 0:
            return PositionSize(
                ticker=ticker,
                direction=direction,
                raw_kelly_fraction=0.0,
                position_pct=0.0,
                position_value=0.0,
                regime=regime,
            )

        # Kelly fraction (half-Kelly)
        raw_kelly = self.kelly_fraction(win_rate, avg_win, avg_loss)
        half_kelly = raw_kelly * self.kelly_multiplier

        # Scale by signal strength
        position_pct = half_kelly * strength * 100  # as % of NAV

        # Apply regime limits
        regime_limits = _REGIME_LIMITS.get(regime, _REGIME_LIMITS["NORMAL"])
        regime_max = regime_limits["max_position_pct"]
        capped = False

        if position_pct > regime_max:
            position_pct = regime_max
            capped = True

        # Global position cap
        if position_pct > self.max_position_pct:
            position_pct = self.max_position_pct
            capped = True

        # Check remaining exposure headroom
        exposure_limit = regime_limits["max_total_exposure_pct"]
        current_gross = current_exposure.get("gross_pct", 0.0)
        headroom = max(0.0, exposure_limit - current_gross)

        if position_pct > headroom:
            position_pct = headroom
            capped = True

        # Dollar value
        position_value = portfolio_value * position_pct / 100.0

        # Small account constraints ($5,444)
        if portfolio_value < 2000:
            if position_value > self.max_equity_position:
                position_value = self.max_equity_position
                position_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0.0
                capped = True

            if position_value > self.max_risk_per_trade:
                position_value = self.max_risk_per_trade
                position_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0.0
                capped = True

        return PositionSize(
            ticker=ticker,
            direction=direction,
            raw_kelly_fraction=round(raw_kelly, 6),
            position_pct=round(position_pct, 4),
            position_value=round(position_value, 2),
            regime=regime,
            capped=capped,
        )

    def check_portfolio_risk(
        self,
        positions: List[Dict],
        portfolio_value: float,
        daily_pnl_pct: float = 0.0,
        drawdown_pct: float = 0.0,
    ) -> RiskReport:
        """Assess portfolio-level risk and flag warnings/violations.

        Parameters
        ----------
        positions : list[dict]
            Each dict: ``{'ticker', 'direction', 'value', 'sector'}``.
        portfolio_value : float
            Current NAV.
        daily_pnl_pct : float
            Today's P&L as % of NAV (negative = loss).
        drawdown_pct : float
            Current drawdown from peak as % (positive number).

        Returns
        -------
        RiskReport
            Portfolio risk metrics, warnings, and violations.
        """
        warnings = []
        violations = []

        total_long = 0.0
        total_short = 0.0
        sector_exp: Dict[str, float] = {}
        max_single = 0.0

        for pos in positions:
            value = abs(float(pos.get("value", 0)))
            pct = (value / portfolio_value * 100) if portfolio_value > 0 else 0.0
            direction = pos.get("direction", "LONG")
            sector = pos.get("sector", "UNKNOWN")

            if direction == "LONG":
                total_long += value
            elif direction == "SHORT":
                total_short += value

            sector_exp[sector] = sector_exp.get(sector, 0.0) + value
            max_single = max(max_single, pct)

        gross = total_long + total_short
        net = total_long - total_short

        long_pct = (total_long / portfolio_value * 100) if portfolio_value > 0 else 0.0
        short_pct = (total_short / portfolio_value * 100) if portfolio_value > 0 else 0.0
        gross_pct = (gross / portfolio_value * 100) if portfolio_value > 0 else 0.0

        # Sector exposure as %
        sector_pct = {
            s: (v / portfolio_value * 100) if portfolio_value > 0 else 0.0
            for s, v in sector_exp.items()
        }

        # --- Exposure checks ---
        if max_single > self.max_position_pct:
            violations.append(
                f"POSITION_CONCENTRATION: {max_single:.1f}% > {self.max_position_pct}%"
            )

        if long_pct > self.max_long_exposure_pct:
            violations.append(
                f"LONG_EXPOSURE: {long_pct:.1f}% > {self.max_long_exposure_pct}%"
            )

        if short_pct > self.max_short_exposure_pct:
            violations.append(
                f"SHORT_EXPOSURE: {short_pct:.1f}% > {self.max_short_exposure_pct}%"
            )

        if gross_pct > self.max_gross_exposure_pct:
            violations.append(
                f"GROSS_EXPOSURE: {gross_pct:.1f}% > {self.max_gross_exposure_pct}%"
            )

        for sector, spct in sector_pct.items():
            if spct > self.max_sector_exposure_pct:
                violations.append(
                    f"SECTOR_EXPOSURE ({sector}): {spct:.1f}% > {self.max_sector_exposure_pct}%"
                )

        # --- Kill switch checks ---
        if daily_pnl_pct <= -self.daily_loss_flatten_pct:
            violations.append(
                f"DAILY_LOSS_FLATTEN: {daily_pnl_pct:.1f}% <= -{self.daily_loss_flatten_pct}%"
            )
        elif daily_pnl_pct <= -self.daily_loss_reduce_pct:
            warnings.append(
                f"DAILY_LOSS_REDUCE: {daily_pnl_pct:.1f}% <= -{self.daily_loss_reduce_pct}%"
            )

        if drawdown_pct >= self.max_drawdown_halt_pct:
            violations.append(
                f"MAX_DRAWDOWN_HALT: {drawdown_pct:.1f}% >= {self.max_drawdown_halt_pct}%"
            )

        return RiskReport(
            total_long_exposure=round(total_long, 2),
            total_short_exposure=round(total_short, 2),
            gross_exposure=round(gross, 2),
            net_exposure=round(net, 2),
            max_single_position_pct=round(max_single, 2),
            sector_exposures=sector_pct,
            warnings=warnings,
            violations=violations,
        )
