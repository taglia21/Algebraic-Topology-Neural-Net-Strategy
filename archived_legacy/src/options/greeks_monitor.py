"""
Portfolio Greeks Monitor
========================

Estimates aggregate portfolio Greeks (delta, gamma, theta, vega) from
live Alpaca option positions and enforces hard limits so the engine
never accumulates excessive directional or volatility exposure.

Estimates are intentionally rough — they only need to be directionally
correct to act as a safety net.  We DO NOT call an option pricing model
on each cycle because that would add latency and API calls.

Example::

    monitor = PortfolioGreeksMonitor()
    greeks = monitor.get_portfolio_greeks(trading_client)
    ok, violations = monitor.is_within_limits(greeks)
    monitor.log_greeks(greeks)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# HARD LIMITS
# ============================================================================

MAX_PORTFOLIO_DELTA = 300     # shares-equivalent net delta
MAX_PORTFOLIO_THETA = -100.0  # max $100/day theta bleed (negative = paying)
MAX_PORTFOLIO_VEGA  = 1000.0  # max vega exposure ($ per 1-pt IV move)


@dataclass
class PortfolioGreeks:
    """Snapshot of estimated portfolio-level Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    num_positions: int = 0


@dataclass
class GreeksViolation:
    """Description of a single limits violation."""
    greek: str
    value: float
    limit: float
    message: str = ""


class PortfolioGreeksMonitor:
    """Query Alpaca positions and estimate aggregate Greeks."""

    # ── Rough per-contract estimates ───────────────────────────────
    #   These are *intentionally* conservative heuristics.
    #   Calls: delta~+50, gamma~+2, theta~-5, vega~+10  (per contract)
    #   Puts:  delta~-50, gamma~+2, theta~-5, vega~+10
    #   Qty sign from Alpaca encodes long (+) vs short (-).
    CALL_DELTA_PER_CONTRACT = 50.0
    PUT_DELTA_PER_CONTRACT  = -50.0
    GAMMA_PER_CONTRACT      = 2.0     # always positive for long
    THETA_PER_CONTRACT      = -5.0    # longs bleed theta
    VEGA_PER_CONTRACT       = 10.0    # longs gain from IV rise

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_greeks: PortfolioGreeks = PortfolioGreeks()

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def get_portfolio_greeks(self, trading_client) -> PortfolioGreeks:
        """
        Query Alpaca positions and compute aggregate Greeks.

        Parameters
        ----------
        trading_client : alpaca.trading.TradingClient
            An authenticated Alpaca TradingClient instance.

        Returns
        -------
        PortfolioGreeks
        """
        greeks = PortfolioGreeks()

        try:
            positions = trading_client.get_all_positions()
        except Exception as exc:
            self.logger.warning(f"Could not fetch positions for Greeks: {exc}")
            self.last_greeks = greeks
            return greeks

        for pos in positions:
            sym = pos.symbol or ""
            # Option OCC symbols are >6 chars with digits early on
            if len(sym) <= 6:
                continue  # equity position — skip

            qty = float(pos.qty) if pos.qty else 0.0
            if qty == 0:
                continue

            is_put = self._is_put(sym)
            delta_per = self.PUT_DELTA_PER_CONTRACT if is_put else self.CALL_DELTA_PER_CONTRACT

            # qty is already signed (+long, -short) from Alpaca
            greeks.delta += qty * delta_per
            greeks.gamma += abs(qty) * self.GAMMA_PER_CONTRACT
            greeks.theta += qty * self.THETA_PER_CONTRACT
            greeks.vega  += qty * self.VEGA_PER_CONTRACT
            greeks.num_positions += 1

        self.last_greeks = greeks
        return greeks

    def is_within_limits(
        self, greeks: PortfolioGreeks | None = None
    ) -> Tuple[bool, List[GreeksViolation]]:
        """
        Check whether *greeks* (default: last computed) are within limits.

        Returns
        -------
        (ok, violations) where ok is True if no limits are breached.
        """
        if greeks is None:
            greeks = self.last_greeks

        violations: List[GreeksViolation] = []

        if abs(greeks.delta) > MAX_PORTFOLIO_DELTA:
            violations.append(GreeksViolation(
                greek="delta",
                value=greeks.delta,
                limit=MAX_PORTFOLIO_DELTA,
                message=f"Delta {greeks.delta:+.0f} exceeds ±{MAX_PORTFOLIO_DELTA}",
            ))

        if greeks.theta < MAX_PORTFOLIO_THETA:
            violations.append(GreeksViolation(
                greek="theta",
                value=greeks.theta,
                limit=MAX_PORTFOLIO_THETA,
                message=f"Theta {greeks.theta:+.1f} exceeds bleed limit {MAX_PORTFOLIO_THETA}",
            ))

        if greeks.vega > MAX_PORTFOLIO_VEGA:
            violations.append(GreeksViolation(
                greek="vega",
                value=greeks.vega,
                limit=MAX_PORTFOLIO_VEGA,
                message=f"Vega {greeks.vega:+.1f} exceeds limit {MAX_PORTFOLIO_VEGA}",
            ))

        ok = len(violations) == 0
        return ok, violations

    def log_greeks(self, greeks: PortfolioGreeks | None = None) -> None:
        """Pretty-print Greeks to the logger."""
        g = greeks or self.last_greeks
        self.logger.info(
            f"📊 Portfolio Greeks — Δ={g.delta:+.0f}  Γ={g.gamma:.0f}  "
            f"Θ={g.theta:+.1f}/day  V={g.vega:+.0f}  "
            f"({g.num_positions} option positions)"
        )

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_put(occ_symbol: str) -> bool:
        """Return True if OCC symbol represents a put option."""
        # OCC format: AAPL260320P00230000 — find the P/C flag
        # Skip underlying letters, then 6-digit date, then P or C
        for i, ch in enumerate(occ_symbol):
            if ch.isdigit():
                # i is start of date; P/C is at i+6
                flag_idx = i + 6
                if flag_idx < len(occ_symbol):
                    return occ_symbol[flag_idx] == "P"
                break
        return False
