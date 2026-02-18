"""
Greeks Manager — Portfolio-Level Greeks Calculation & Monitoring
================================================================

Calculates and aggregates option Greeks (delta, gamma, theta, vega)
across the entire options portfolio.  Provides real-time exposure
snapshots and determines when hedging is required.

Key capabilities:
  • Per-position & portfolio-level Greeks via Black-Scholes
  • Delta exposure monitoring with configurable bands
  • Gamma scalping detection (high-gamma + high-realized-vol)
  • Circuit-breaker flags for excessive gamma/vega exposure
  • Theta decay tracking for income attribution

Usage:
    from src.greeks_manager import GreeksManager, GreeksConfig

    gm = GreeksManager()
    snapshot = gm.calculate_portfolio_greeks(positions, prices)
    if snapshot.needs_hedge:
        print(f"Hedge needed: portfolio delta = {snapshot.net_delta:.2f}")
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GreeksConfig:
    """Configurable parameters for Greeks calculation & monitoring."""
    # Delta hedging bands
    max_portfolio_delta: float = 0.50          # Max absolute delta per $100k equity
    delta_hedge_threshold: float = 0.30        # Trigger hedge when |delta| exceeds

    # Gamma / Vega limits (circuit breaker thresholds)
    max_portfolio_gamma: float = 0.10          # Max absolute portfolio gamma
    max_portfolio_vega: float = 1000.0         # Max absolute portfolio vega ($)

    # Gamma scalping
    gamma_scalp_min_gamma: float = 0.05        # Min gamma to consider scalping
    gamma_scalp_min_realized_vol: float = 0.02 # Min daily realized vol (2%)

    # Black-Scholes parameters
    risk_free_rate: float = 0.05               # Annual risk-free rate
    default_iv: float = 0.25                   # Fallback IV when unavailable

    # Theta tracking
    theta_warning_daily: float = -500.0        # Warn if daily theta exceeds (credit sellers)

    # Scaling
    notional_scale: float = 100_000.0          # Greeks normalized per $100k equity


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class OptionSide(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class PositionGreeks:
    """Greeks for a single options position."""
    symbol: str
    underlying: str
    option_type: OptionSide
    strike: float
    expiration_date: date
    quantity: int            # +long / -short
    spot_price: float
    iv: float
    # Greeks (per-contract, already scaled by quantity * 100)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0      # daily theta ($)
    vega: float = 0.0       # $ change per 1% IV move
    rho: float = 0.0

    @property
    def time_to_expiry(self) -> float:
        """Years to expiration (min 1 day)."""
        days = max(1, (self.expiration_date - date.today()).days)
        return days / 365.0

    @property
    def dte(self) -> int:
        return max(0, (self.expiration_date - date.today()).days)


@dataclass
class PortfolioGreeksSnapshot:
    """Aggregated portfolio-level Greeks snapshot."""
    timestamp: datetime
    # Aggregated Greeks (sum across all positions)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0          # daily theta ($)
    net_vega: float = 0.0           # $ per 1% IV move
    net_rho: float = 0.0

    # Per-underlying breakdown
    delta_by_underlying: Dict[str, float] = field(default_factory=dict)

    # Position details
    position_greeks: List[PositionGreeks] = field(default_factory=list)
    position_count: int = 0

    # Flags
    needs_hedge: bool = False
    hedge_reason: str = ""
    gamma_scalp_opportunity: bool = False
    circuit_breaker_triggered: bool = False
    circuit_breaker_reason: str = ""

    # Thresholds used
    delta_threshold_used: float = 0.0
    max_delta_used: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "net_delta": round(self.net_delta, 4),
            "net_gamma": round(self.net_gamma, 4),
            "net_theta": round(self.net_theta, 2),
            "net_vega": round(self.net_vega, 2),
            "net_rho": round(self.net_rho, 4),
            "delta_by_underlying": {
                k: round(v, 4) for k, v in self.delta_by_underlying.items()
            },
            "position_count": self.position_count,
            "needs_hedge": self.needs_hedge,
            "hedge_reason": self.hedge_reason,
            "gamma_scalp_opportunity": self.gamma_scalp_opportunity,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "circuit_breaker_reason": self.circuit_breaker_reason,
        }


# ============================================================================
# BLACK-SCHOLES GREEKS ENGINE
# ============================================================================

class BSGreeks:
    """Black-Scholes Greeks calculator (static methods)."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return BSGreeks.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionSide) -> float:
        """Per-share delta (-1 to +1)."""
        if T <= 0:
            if option_type == OptionSide.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BSGreeks.d1(S, K, T, r, sigma)
        if option_type == OptionSide.CALL:
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1.0)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Per-share gamma (same for calls/puts)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = BSGreeks.d1(S, K, T, r, sigma)
        return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionSide) -> float:
        """Per-share theta ($ per day)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BSGreeks.d1(S, K, T, r, sigma)
        d2 = BSGreeks.d2(S, K, T, r, sigma)
        term1 = -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        if option_type == OptionSide.CALL:
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        return float((term1 + term2) / 365.0)

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Per-share vega ($ change for 1% IV move)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BSGreeks.d1(S, K, T, r, sigma)
        return float(S * norm.pdf(d1) * math.sqrt(T) / 100.0)

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float,
            option_type: OptionSide) -> float:
        """Per-share rho ($ change for 1% rate move)."""
        if T <= 0:
            return 0.0
        d2 = BSGreeks.d2(S, K, T, r, sigma)
        if option_type == OptionSide.CALL:
            return float(K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0)
        else:
            return float(-K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0)


# ============================================================================
# GREEKS MANAGER
# ============================================================================

class GreeksManager:
    """
    Portfolio-level Greeks aggregation and monitoring.

    Computes Greeks for each options position using Black-Scholes,
    then aggregates to portfolio level and checks exposure limits.
    """

    def __init__(self, config: Optional[GreeksConfig] = None):
        self.config = config or GreeksConfig()
        self._last_snapshot: Optional[PortfolioGreeksSnapshot] = None
        self._snapshot_history: List[PortfolioGreeksSnapshot] = []
        self._max_history = 500
        logger.info("GreeksManager initialized")

    # ── Core calculation ────────────────────────────────────────────

    def calculate_position_greeks(
        self,
        symbol: str,
        underlying: str,
        option_type: str,       # "call" or "put"
        strike: float,
        expiration: str,        # "YYYY-MM-DD"
        quantity: int,          # +long / -short
        spot_price: float,
        iv: Optional[float] = None,
    ) -> PositionGreeks:
        """
        Calculate Greeks for a single options position.

        Args:
            symbol: Option contract symbol
            underlying: Underlying ticker
            option_type: "call" or "put"
            strike: Strike price
            expiration: Expiration date string (YYYY-MM-DD)
            quantity: Number of contracts (+long, -short)
            spot_price: Current underlying price
            iv: Implied volatility (decimal, e.g. 0.25 = 25%)

        Returns:
            PositionGreeks with per-position (quantity * 100 multiplied) values
        """
        opt_side = OptionSide.CALL if option_type.lower() == "call" else OptionSide.PUT

        # Parse expiration
        try:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            exp_date = date.today()

        T = max(1, (exp_date - date.today()).days) / 365.0
        sigma = iv if iv and iv > 0 else self.config.default_iv
        r = self.config.risk_free_rate
        S = spot_price
        K = strike

        # Per-share Greeks
        d = BSGreeks.delta(S, K, T, r, sigma, opt_side)
        g = BSGreeks.gamma(S, K, T, r, sigma)
        t = BSGreeks.theta(S, K, T, r, sigma, opt_side)
        v = BSGreeks.vega(S, K, T, r, sigma)
        rh = BSGreeks.rho(S, K, T, r, sigma, opt_side)

        # Scale by quantity * 100 (each contract = 100 shares)
        multiplier = quantity * 100

        return PositionGreeks(
            symbol=symbol,
            underlying=underlying,
            option_type=opt_side,
            strike=strike,
            expiration_date=exp_date,
            quantity=quantity,
            spot_price=spot_price,
            iv=sigma,
            delta=d * multiplier,
            gamma=g * multiplier,
            theta=t * multiplier,
            vega=v * multiplier,
            rho=rh * multiplier,
        )

    def calculate_portfolio_greeks(
        self,
        positions: List[dict],
        prices: Optional[Dict[str, float]] = None,
    ) -> PortfolioGreeksSnapshot:
        """
        Aggregate Greeks across all option positions.

        Args:
            positions: List of position dicts, each with keys:
                - symbol: option contract symbol
                - underlying: underlying ticker
                - option_type: "call" or "put"
                - strike: float
                - expiration: "YYYY-MM-DD"
                - quantity: int (+long / -short)
                - spot_price: float (or looked up from prices)
                - iv: float (optional, decimal)
            prices: Optional dict of underlying -> current price

        Returns:
            PortfolioGreeksSnapshot with aggregated values & flags
        """
        snapshot = PortfolioGreeksSnapshot(
            timestamp=datetime.now(),
            delta_threshold_used=self.config.delta_hedge_threshold,
            max_delta_used=self.config.max_portfolio_delta,
        )

        if not positions:
            self._last_snapshot = snapshot
            return snapshot

        position_greeks: List[PositionGreeks] = []
        delta_by_ul: Dict[str, float] = {}

        for pos in positions:
            try:
                # Resolve spot price
                spot = pos.get("spot_price", 0.0)
                if (not spot or spot <= 0) and prices:
                    spot = prices.get(pos.get("underlying", ""), 0.0)
                if spot <= 0:
                    logger.debug(f"No spot price for {pos.get('symbol')} — skipping")
                    continue

                pg = self.calculate_position_greeks(
                    symbol=pos.get("symbol", ""),
                    underlying=pos.get("underlying", ""),
                    option_type=pos.get("option_type", "call"),
                    strike=float(pos.get("strike", 0)),
                    expiration=pos.get("expiration", ""),
                    quantity=int(pos.get("quantity", 0)),
                    spot_price=spot,
                    iv=pos.get("iv"),
                )
                position_greeks.append(pg)

                # Accumulate per-underlying delta
                ul = pg.underlying
                delta_by_ul[ul] = delta_by_ul.get(ul, 0.0) + pg.delta

            except Exception as e:
                logger.warning(f"Greeks calc failed for {pos}: {e}")
                continue

        # Aggregate
        snapshot.position_greeks = position_greeks
        snapshot.position_count = len(position_greeks)
        snapshot.delta_by_underlying = delta_by_ul
        snapshot.net_delta = sum(pg.delta for pg in position_greeks)
        snapshot.net_gamma = sum(pg.gamma for pg in position_greeks)
        snapshot.net_theta = sum(pg.theta for pg in position_greeks)
        snapshot.net_vega = sum(pg.vega for pg in position_greeks)
        snapshot.net_rho = sum(pg.rho for pg in position_greeks)

        # ── Check hedge requirement ──
        self._evaluate_hedge_need(snapshot)

        # ── Check circuit breakers ──
        self._evaluate_circuit_breakers(snapshot)

        # ── Check gamma scalping ──
        self._evaluate_gamma_scalp(snapshot)

        # Store snapshot
        self._last_snapshot = snapshot
        self._snapshot_history.append(snapshot)
        if len(self._snapshot_history) > self._max_history:
            self._snapshot_history = self._snapshot_history[-self._max_history:]

        return snapshot

    # ── Exposure queries ────────────────────────────────────────────

    def get_delta_exposure(self) -> Dict[str, float]:
        """
        Return current delta exposure per underlying.

        Returns:
            Dict mapping underlying -> net delta
        """
        if self._last_snapshot is None:
            return {}
        return dict(self._last_snapshot.delta_by_underlying)

    def get_portfolio_delta(self) -> float:
        """Return total portfolio delta."""
        if self._last_snapshot is None:
            return 0.0
        return self._last_snapshot.net_delta

    def get_portfolio_summary(self) -> dict:
        """Human-readable portfolio Greeks summary."""
        if self._last_snapshot is None:
            return {"status": "no_data"}
        s = self._last_snapshot
        return {
            "delta": round(s.net_delta, 2),
            "gamma": round(s.net_gamma, 4),
            "theta_daily": round(s.net_theta, 2),
            "vega": round(s.net_vega, 2),
            "positions": s.position_count,
            "needs_hedge": s.needs_hedge,
            "hedge_reason": s.hedge_reason,
            "circuit_breaker": s.circuit_breaker_triggered,
        }

    def needs_hedge(self) -> Tuple[bool, str]:
        """
        Check whether portfolio exceeds delta hedge threshold.

        Returns:
            (needs_hedge: bool, reason: str)
        """
        if self._last_snapshot is None:
            return False, "no snapshot"
        return self._last_snapshot.needs_hedge, self._last_snapshot.hedge_reason

    def is_circuit_breaker_triggered(self) -> Tuple[bool, str]:
        """
        Check if gamma or vega exposure triggers circuit breaker.

        Returns:
            (triggered: bool, reason: str)
        """
        if self._last_snapshot is None:
            return False, "no snapshot"
        return (
            self._last_snapshot.circuit_breaker_triggered,
            self._last_snapshot.circuit_breaker_reason,
        )

    def has_gamma_scalp_opportunity(self) -> bool:
        """Return True if current gamma is high enough for scalping."""
        if self._last_snapshot is None:
            return False
        return self._last_snapshot.gamma_scalp_opportunity

    # ── Snapshot history ────────────────────────────────────────────

    def get_last_snapshot(self) -> Optional[PortfolioGreeksSnapshot]:
        return self._last_snapshot

    def get_snapshot_history(self) -> List[PortfolioGreeksSnapshot]:
        return list(self._snapshot_history)

    # ── Internal evaluation ─────────────────────────────────────────

    def _evaluate_hedge_need(self, snapshot: PortfolioGreeksSnapshot):
        """Set hedge flags based on delta exposure vs config thresholds."""
        abs_delta = abs(snapshot.net_delta)

        if abs_delta > self.config.max_portfolio_delta:
            snapshot.needs_hedge = True
            snapshot.hedge_reason = (
                f"Portfolio delta {snapshot.net_delta:+.2f} exceeds "
                f"max {self.config.max_portfolio_delta:.2f}"
            )
        elif abs_delta > self.config.delta_hedge_threshold:
            snapshot.needs_hedge = True
            snapshot.hedge_reason = (
                f"Portfolio delta {snapshot.net_delta:+.2f} exceeds "
                f"hedge threshold {self.config.delta_hedge_threshold:.2f}"
            )
        else:
            snapshot.needs_hedge = False
            snapshot.hedge_reason = ""

    def _evaluate_circuit_breakers(self, snapshot: PortfolioGreeksSnapshot):
        """Check gamma & vega against circuit-breaker limits."""
        reasons = []

        if abs(snapshot.net_gamma) > self.config.max_portfolio_gamma:
            reasons.append(
                f"Gamma {snapshot.net_gamma:+.4f} exceeds "
                f"max {self.config.max_portfolio_gamma:.4f}"
            )

        if abs(snapshot.net_vega) > self.config.max_portfolio_vega:
            reasons.append(
                f"Vega ${snapshot.net_vega:+.0f} exceeds "
                f"max ${self.config.max_portfolio_vega:.0f}"
            )

        if reasons:
            snapshot.circuit_breaker_triggered = True
            snapshot.circuit_breaker_reason = "; ".join(reasons)
        else:
            snapshot.circuit_breaker_triggered = False
            snapshot.circuit_breaker_reason = ""

    def _evaluate_gamma_scalp(self, snapshot: PortfolioGreeksSnapshot,
                              realized_vol: Optional[float] = None):
        """
        Detect gamma scalping opportunity.

        Gamma scalping profits when:
          1. Portfolio has significant LONG gamma (positive gamma)
          2. Realized volatility is high enough to offset theta
        """
        if snapshot.net_gamma <= 0:
            # Short gamma — no scalping opportunity
            snapshot.gamma_scalp_opportunity = False
            return

        if snapshot.net_gamma >= self.config.gamma_scalp_min_gamma:
            # If we don't have realized vol, set flag based on gamma alone
            if realized_vol is None or realized_vol >= self.config.gamma_scalp_min_realized_vol:
                snapshot.gamma_scalp_opportunity = True
            else:
                snapshot.gamma_scalp_opportunity = False
        else:
            snapshot.gamma_scalp_opportunity = False


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gm = GreeksManager()

    # Simulate a portfolio with 2 positions
    positions = [
        {
            "symbol": "SPY250321C00500000",
            "underlying": "SPY",
            "option_type": "call",
            "strike": 500.0,
            "expiration": "2025-03-21",
            "quantity": 5,    # long 5 calls
            "spot_price": 495.0,
            "iv": 0.18,
        },
        {
            "symbol": "SPY250321P00490000",
            "underlying": "SPY",
            "option_type": "put",
            "strike": 490.0,
            "expiration": "2025-03-21",
            "quantity": -3,   # short 3 puts
            "spot_price": 495.0,
            "iv": 0.20,
        },
    ]

    snapshot = gm.calculate_portfolio_greeks(positions)
    print(f"Portfolio Greeks:")
    print(f"  Delta: {snapshot.net_delta:+.2f}")
    print(f"  Gamma: {snapshot.net_gamma:+.4f}")
    print(f"  Theta: ${snapshot.net_theta:+.2f}/day")
    print(f"  Vega:  ${snapshot.net_vega:+.2f}/1% IV")
    print(f"  Needs hedge: {snapshot.needs_hedge} — {snapshot.hedge_reason}")
    print(f"  Circuit breaker: {snapshot.circuit_breaker_triggered}")
    print(f"  Gamma scalp: {snapshot.gamma_scalp_opportunity}")
    print("Greeks manager OK")


# ============================================================================
# PORTFOLIO GREEKS AGGREGATOR — wraps GreeksManager + DeltaHedger
# ============================================================================

@dataclass
class AggregatedExposure:
    """Full portfolio exposure: options Greeks + equity beta-delta."""
    options_delta: float = 0.0
    equity_delta: float = 0.0       # sum(shares × beta × share_price / scale)
    total_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    needs_hedge: bool = False
    hedge_reason: str = ""
    circuit_breaker: bool = False
    circuit_breaker_reason: str = ""
    gamma_scalp: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def describe(self) -> str:
        return (
            f"Δ_total={self.total_delta:+.2f} "
            f"(opt={self.options_delta:+.2f} eq={self.equity_delta:+.2f}) "
            f"Γ={self.net_gamma:+.4f} Θ=${self.net_theta:+.1f}/d "
            f"ν=${self.net_vega:+.1f} "
            f"hedge={'YES' if self.needs_hedge else 'no'}"
        )


class PortfolioGreeksAggregator:
    """
    Aggregate options Greeks and equity beta-delta into a single exposure.

    Combines:
      1. GreeksManager  — BS options Greeks (delta, gamma, theta, vega)
      2. Equity holdings — approximated as delta-1 × beta per share
      3. Auto-hedge decision — triggers DeltaHedger when thresholds exceeded

    Parameters
    ----------
    greeks_config : GreeksConfig, optional
    equity_beta_map : dict, optional
        {symbol: beta_to_SPY}. Defaults to 1.0 for unknowns.
    notional_scale : float
        Scale for delta normalisation (default $100k).
    """

    DEFAULT_BETAS: Dict[str, float] = {
        "SPY": 1.0, "QQQ": 1.15, "IWM": 1.20,
        "XLK": 1.10, "XLF": 1.05, "XLE": 0.95,
        "XLV": 0.80, "XLP": 0.65, "XLI": 1.00,
        "AAPL": 1.20, "MSFT": 1.10, "GOOGL": 1.15,
        "AMZN": 1.25, "NVDA": 1.60, "META": 1.30,
        "TSLA": 1.80, "JPM": 1.10, "BAC": 1.15, "GS": 1.25,
        "TLT": -0.30, "GLD": 0.05, "HYG": 0.40,
    }

    def __init__(
        self,
        greeks_config: Optional[GreeksConfig] = None,
        equity_beta_map: Optional[Dict[str, float]] = None,
        notional_scale: float = 100_000.0,
    ):
        self.greeks_mgr = GreeksManager(config=greeks_config)
        self.betas = dict(self.DEFAULT_BETAS)
        if equity_beta_map:
            self.betas.update(equity_beta_map)
        self.scale = notional_scale
        self._last_exposure: Optional[AggregatedExposure] = None
        self.logger = logging.getLogger(f"{__name__}.PortfolioGreeksAggregator")

    def get_beta(self, symbol: str) -> float:
        """Look up beta for a symbol (default 1.0)."""
        return self.betas.get(symbol, 1.0)

    def compute_equity_delta(
        self, equity_positions: Dict[str, Tuple[int, float]]
    ) -> float:
        """
        Compute equity beta-delta.

        Parameters
        ----------
        equity_positions : dict
            {symbol: (shares, current_price)}

        Returns
        -------
        float
            Equity delta normalised to ``notional_scale``.
        """
        raw_delta = 0.0
        for sym, (shares, price) in equity_positions.items():
            beta = self.get_beta(sym)
            raw_delta += shares * price * beta
        return raw_delta / self.scale

    def aggregate(
        self,
        option_positions: Optional[List[dict]] = None,
        equity_positions: Optional[Dict[str, Tuple[int, float]]] = None,
    ) -> AggregatedExposure:
        """
        Compute aggregated exposure for full portfolio.

        Parameters
        ----------
        option_positions : list[dict], optional
            Option positions for GreeksManager.calculate_portfolio_greeks().
        equity_positions : dict, optional
            {symbol: (shares, current_price)} for equity delta.

        Returns
        -------
        AggregatedExposure
        """
        exp = AggregatedExposure()

        # Options Greeks
        if option_positions:
            snapshot = self.greeks_mgr.calculate_portfolio_greeks(option_positions)
            exp.options_delta = snapshot.net_delta
            exp.net_gamma = snapshot.net_gamma
            exp.net_theta = snapshot.net_theta
            exp.net_vega = snapshot.net_vega
            exp.circuit_breaker = snapshot.circuit_breaker_triggered
            exp.circuit_breaker_reason = snapshot.circuit_breaker_reason
            exp.gamma_scalp = snapshot.gamma_scalp_opportunity

        # Equity delta
        if equity_positions:
            exp.equity_delta = self.compute_equity_delta(equity_positions)

        # Total
        exp.total_delta = exp.options_delta + exp.equity_delta

        # Hedge decision on total delta
        threshold = self.greeks_mgr.config.delta_hedge_threshold
        max_delta = self.greeks_mgr.config.max_portfolio_delta
        abs_td = abs(exp.total_delta)
        if abs_td > max_delta:
            exp.needs_hedge = True
            exp.hedge_reason = (
                f"Total delta {exp.total_delta:+.2f} exceeds max {max_delta:.2f}"
            )
        elif abs_td > threshold:
            exp.needs_hedge = True
            exp.hedge_reason = (
                f"Total delta {exp.total_delta:+.2f} exceeds threshold {threshold:.2f}"
            )

        self._last_exposure = exp
        self.logger.info(f"Aggregated exposure: {exp.describe()}")
        return exp

    @property
    def last_exposure(self) -> Optional[AggregatedExposure]:
        return self._last_exposure
