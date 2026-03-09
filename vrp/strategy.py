"""
vrp/strategy.py
===============
VRP Strategy Engine — Systematic SPX Put Credit Spread Trading.

This module implements the core strategy logic:
1. Signal generation (should we open a new spread?)
2. Strike selection (which strikes to sell/buy?)
3. Position management (when to close, roll, or stop out?)

The strategy harvests the Volatility Risk Premium: the persistent tendency
for implied volatility to exceed realized volatility. By selling OTM put
spreads, we collect this premium while maintaining defined risk.

Key design principles:
- Every trade has defined max loss (spread width - credit)
- Position sizing is risk-based, not notional-based
- VIX regime determines when and how much to trade
- No holding to expiry — always manage before last week
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from vrp.config import Config, SpreadConfig, VIXRegimeConfig
from vrp.utils import (
    Greeks, bs_greeks, bs_put_price, implied_vol,
    dte, years_to_expiry, next_monthly_expiry,
)
from vrp.signals import SignalState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TradeAction(Enum):
    """Actions the strategy can recommend."""
    OPEN = "open"
    CLOSE_PROFIT = "close_profit"
    CLOSE_STOP = "close_stop"
    CLOSE_EXPIRY = "close_expiry"
    ROLL = "roll"
    HOLD = "hold"


class VIXRegime(Enum):
    """VIX-based market regime classification.

    Regime boundaries (from config):
        TOO_LOW:   VIX < min_vix (16) — no edge, skip
        LOW:       VIX 16-20 — VRP exists but thin, reduced sizing (0.35x)
        STANDARD:  VIX 20-25 — sweet spot, full sizing
        ELEVATED:  VIX 25-35 — rich premium, capped sizing (0.75x)
                   Sub-zone 25-27: danger zone (panic transition), 0.375x
        CRISIS:    VIX > 35 — tail risk, no new trades

    Alpha experiment evidence (18 configs, 2020-2025):
        Lowering the floor from 20 to 16 captured 259 additional trades
        (437 total vs 178 baseline) while maintaining 77.8% win rate.
        All 6 years profitable vs 2 losing years at VIX 20 floor.
        The VRP at VIX 16-20 is thinner but persistent (IV/RV ~1.3x).
    """
    TOO_LOW = "too_low"       # VIX < 16: no edge, premium too thin
    LOW = "low"               # VIX 16-20: VRP exists, trade at 0.35x sizing
    STANDARD = "standard"     # VIX 20-25: full-size sweet spot
    ELEVATED = "elevated"     # VIX 25-35: rich premium, capped sizing
    CRISIS = "crisis"         # VIX > 35: stay out


@dataclass
class SpreadLeg:
    """One leg of an option spread."""
    strike: float
    expiry: date
    option_type: str = "put"  # always put for our strategy
    side: str = "sell"        # "sell" for short leg, "buy" for long leg
    quantity: int = 1
    premium: float = 0.0     # price per contract (in dollars, not per-share)
    greeks: Optional[Greeks] = None


@dataclass
class SpreadPosition:
    """A complete put credit spread position."""
    id: str                          # unique position identifier
    short_leg: SpreadLeg             # the sold put (higher strike)
    long_leg: SpreadLeg              # the bought put (lower strike)
    entry_date: date                 # when opened
    entry_credit: float              # net credit received (per spread, in dollars)
    quantity: int = 1                # number of spread contracts
    current_value: float = 0.0       # current mark-to-market cost to close
    spx_at_entry: float = 0.0       # SPX level when opened
    vix_at_entry: float = 0.0       # VIX level when opened
    status: str = "open"             # "open", "closed", "rolled"
    close_date: Optional[date] = None
    close_pnl: float = 0.0          # realized P&L
    close_reason: str = ""           # why we closed

    @property
    def spread_width(self) -> float:
        """Spread width in points."""
        return self.short_leg.strike - self.long_leg.strike

    @property
    def max_risk(self) -> float:
        """Maximum risk per contract in dollars."""
        return (self.spread_width * 100) - self.entry_credit

    @property
    def total_max_risk(self) -> float:
        """Total maximum risk across all contracts."""
        return self.max_risk * self.quantity

    @property
    def current_pnl(self) -> float:
        """Unrealized P&L (positive = profit)."""
        return (self.entry_credit - self.current_value) * self.quantity

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of max credit."""
        if self.entry_credit <= 0:
            return 0.0
        return self.current_pnl / (self.entry_credit * self.quantity)

    @property
    def days_held(self) -> int:
        """Days since entry."""
        today = self.close_date or date.today()
        return (today - self.entry_date).days

    @property
    def dte_remaining(self) -> int:
        """Days to expiration remaining."""
        today = self.close_date or date.today()
        return max(0, (self.short_leg.expiry - today).days)


# ---------------------------------------------------------------------------
# VIX Regime Classifier
# ---------------------------------------------------------------------------

class VIXRegimeClassifier:
    """Classify current VIX level into a trading regime.

    Simple, transparent, parameter-sparse approach. VIX levels are directly
    observable and thresholds are empirically calibrated from 5-year
    per-VIX-point P&L audit:

    - Below 20: zero or negative per-trade edge (VIX 14-18: $10, VIX 19-20: -$18)
    - 20-25: sweet spot ($76-97/trade avg), full sizing
    - 25-27: danger zone (-$128/trade), panic-transition regime, 0.375x sizing
    - 27-35: best returns ($77-213/trade), 0.75x sizing
    - Above 35: regime break risk (2008, 2020, 2025 tariff crash)
    """

    def __init__(self, config: VIXRegimeConfig) -> None:
        self.cfg = config

    def classify(self, vix: float) -> VIXRegime:
        """Classify VIX level into a regime."""
        if vix < self.cfg.min_vix:
            return VIXRegime.TOO_LOW
        elif vix < self.cfg.standard_low:
            return VIXRegime.LOW
        elif vix <= self.cfg.standard_high:
            return VIXRegime.STANDARD
        elif vix <= self.cfg.max_vix:
            return VIXRegime.ELEVATED
        else:
            return VIXRegime.CRISIS

    def sizing_multiplier(self, vix: float) -> float:
        """Return position sizing multiplier based on VIX regime.

        Returns 0.0 for regimes where we don't trade.
        """
        regime = self.classify(vix)
        if regime in (VIXRegime.TOO_LOW, VIXRegime.CRISIS):
            return 0.0
        elif regime == VIXRegime.LOW:
            return self.cfg.low_vol_sizing_mult
        elif regime == VIXRegime.STANDARD:
            return 1.0
        elif regime == VIXRegime.ELEVATED:
            # VIX 25-27 transition zone: halve the elevated multiplier.
            # Audit showed -$128/trade in this band (panic transition).
            # VIX 27+ gets full elevated multiplier.
            if vix < 27.0:
                return self.cfg.elevated_sizing_mult * 0.50
            return self.cfg.elevated_sizing_mult
        return 1.0

    def should_widen_spread(self, vix: float) -> bool:
        """Whether to use wider spreads (more protection) in elevated VIX."""
        return vix > 25.0


# ---------------------------------------------------------------------------
# Strike Selector
# ---------------------------------------------------------------------------

class StrikeSelector:
    """Select optimal strikes for put credit spreads.

    Uses delta-targeting to find the short put strike, then sets the
    long put at a fixed width below. This approach adapts to volatility:
    - In high IV: the same delta is further OTM (more room)
    - In low IV: the same delta is closer to ATM (tighter margin)

    For backtesting, we use Black-Scholes to calculate theoretical deltas.
    In live trading, we use IBKR's real-time greeks from the option chain.
    """

    def __init__(self, config: SpreadConfig) -> None:
        self.cfg = config

    def find_available_expiries(
        self,
        spx_price: float,
        as_of: Optional[date] = None,
    ) -> List[date]:
        """Find all valid monthly expiries within the DTE window."""
        today = as_of or date.today()
        expiries = []
        # Check next 3 months of monthly expiries
        check_date = today
        for _ in range(4):
            exp = next_monthly_expiry(check_date)
            remaining = (exp - today).days
            if self.cfg.min_dte <= remaining <= self.cfg.max_dte:
                expiries.append(exp)
            # Move to next month
            if exp.month == 12:
                check_date = date(exp.year + 1, 1, 1)
            else:
                check_date = date(exp.year, exp.month + 1, 1)
        return expiries

    def find_short_strike(
        self,
        spx_price: float,
        expiry: date,
        iv: float,
        as_of: Optional[date] = None,
        risk_free_rate: float = 0.05,
        available_strikes: Optional[List[float]] = None,
    ) -> float:
        """Find the optimal short put strike targeting the configured delta.

        Parameters
        ----------
        spx_price : Current SPX price
        expiry : Option expiration date
        iv : Implied volatility (annualized)
        as_of : Date to calculate from (default: today)
        risk_free_rate : Risk-free rate for BS
        available_strikes : If provided, select from these strikes only

        Returns
        -------
        Optimal short put strike price
        """
        T = years_to_expiry(expiry, as_of)
        if T <= 0:
            return 0.0

        target_delta = self.cfg.short_delta_target  # e.g., -0.12

        if available_strikes:
            # Find the strike whose delta is closest to our target
            best_strike = 0.0
            best_diff = float('inf')

            for strike in sorted(available_strikes, reverse=True):
                if strike >= spx_price:
                    continue  # skip ITM strikes
                greeks = bs_greeks(spx_price, strike, T, risk_free_rate, iv, "put")
                diff = abs(greeks.delta - target_delta)
                if diff < best_diff:
                    best_diff = diff
                    best_strike = strike

            return best_strike

        # Search in 5-point increments (SPX standard strikes)
        best_strike = 0.0
        best_diff = float('inf')

        # Search range: 2% to 15% below current price
        low = int(spx_price * 0.85)
        high = int(spx_price * 0.98)

        for strike in range(low, high + 1, 5):  # SPX strikes in 5-pt increments
            greeks = bs_greeks(spx_price, float(strike), T, risk_free_rate, iv, "put")
            diff = abs(greeks.delta - target_delta)

            # Also check it's within acceptable range
            if self.cfg.short_delta_min <= greeks.delta <= self.cfg.short_delta_max:
                if diff < best_diff:
                    best_diff = diff
                    best_strike = float(strike)

        # Fallback: if no strike in delta range, use percentage-based
        if best_strike == 0.0:
            # ~8% OTM as fallback
            best_strike = round(spx_price * 0.92 / 5) * 5

        return best_strike

    def build_spread(
        self,
        spx_price: float,
        expiry: date,
        iv: float,
        vix: float,
        as_of: Optional[date] = None,
        risk_free_rate: float = 0.05,
        available_strikes: Optional[List[float]] = None,
    ) -> Optional[Tuple[SpreadLeg, SpreadLeg, float]]:
        """Build a complete put credit spread.

        Returns
        -------
        Tuple of (short_leg, long_leg, net_credit) or None if no valid spread.
        """
        T = years_to_expiry(expiry, as_of)
        if T <= 0:
            return None

        # Dynamic spread width based on account size and VIX
        width = self.cfg.spread_width  # 15 pts default
        # Adjust width to what the account can actually afford
        if hasattr(self, '_account_equity') and self._account_equity > 0:
            # Reserve 20% of equity as buffer — can't risk everything on one spread
            max_affordable = int(self._account_equity * 0.80 / 100)
            width = min(width, max(10, max_affordable))  # floor at 10 points
        # VIX 25-27 danger zone: NARROW spread to reduce max loss.
        # Audit showed -$128/trade here — limit damage by using minimum width.
        if 25 <= vix < 27:
            width = 10  # absolute minimum, caps risk at $1,000/contract
        elif vix >= 27:
            # VIX 27+: widen for richer premium — but only if affordable
            target_width = width + 10  # try 10 pts wider
            if hasattr(self, '_account_equity') and self._account_equity > 0:
                max_affordable = int(self._account_equity * 0.80 / 100)
                target_width = min(target_width, max_affordable)
            width = max(width, min(target_width, 50))  # cap at 50 pts

        short_strike = self.find_short_strike(
            spx_price, expiry, iv, as_of, risk_free_rate, available_strikes
        )
        if short_strike <= 0:
            return None

        long_strike = short_strike - width

        # Calculate theoretical prices
        short_price = bs_put_price(spx_price, short_strike, T, risk_free_rate, iv)
        long_price = bs_put_price(spx_price, long_strike, T, risk_free_rate, iv)
        net_credit = (short_price - long_price) * 100  # per contract in dollars

        # Check minimum credit threshold
        max_risk = width * 100
        if net_credit / max_risk < self.cfg.min_credit_pct:
            logger.debug(
                f"Rejected spread: credit ${net_credit:.0f} / "
                f"risk ${max_risk:.0f} = {net_credit/max_risk:.1%} "
                f"< min {self.cfg.min_credit_pct:.0%}"
            )
            return None

        short_greeks = bs_greeks(spx_price, short_strike, T, risk_free_rate, iv, "put")
        long_greeks = bs_greeks(spx_price, long_strike, T, risk_free_rate, iv, "put")

        short_leg = SpreadLeg(
            strike=short_strike,
            expiry=expiry,
            side="sell",
            premium=short_price * 100,
            greeks=short_greeks,
        )
        long_leg = SpreadLeg(
            strike=long_strike,
            expiry=expiry,
            side="buy",
            premium=long_price * 100,
            greeks=long_greeks,
        )

        return short_leg, long_leg, net_credit


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Manage open spread positions — decide hold/close/roll.

    Implements a systematic exit framework:
    1. Profit target: close at 50% of max credit (or 75% near expiry)
    2. Stop loss: close if spread reaches 2x credit received
    3. Time stop: close 3 days before expiry (no pin risk)
    4. Roll: if ITM near expiry and VIX conditions are favorable
    """

    def __init__(self, config: SpreadConfig) -> None:
        self.cfg = config

    def evaluate(
        self,
        position: SpreadPosition,
        spx_price: float,
        vix: float,
        as_of: Optional[date] = None,
        spx_low: Optional[float] = None,
    ) -> TradeAction:
        """Evaluate an open position and recommend an action.

        Parameters
        ----------
        position : The open spread position
        spx_price : Current SPX close price
        vix : Current VIX level
        as_of : Current date
        spx_low : Intraday low — triggers immediate exit if below short strike

        Returns
        -------
        Recommended action
        """
        today = as_of or date.today()
        remaining_dte = (position.short_leg.expiry - today).days

        # 0. Intraday short-strike breach: if SPX low breached our short
        # strike at any point during the day, exit immediately. This catches
        # flash crashes and sharp selloffs before the end-of-day stop loss
        # triggers. Institutional desks call this a "touch" stop.
        if spx_low is not None and spx_low < position.short_leg.strike:
            logger.info(
                f"Short strike breach: SPX low {spx_low:.0f} < "
                f"short strike {position.short_leg.strike:.0f}"
            )
            return TradeAction.CLOSE_STOP

        # 1. Time stop: force close near expiry
        if remaining_dte <= self.cfg.close_before_expiry_days:
            return TradeAction.CLOSE_EXPIRY

        # 2. Profit target
        pnl_pct = position.pnl_pct
        if remaining_dte <= self.cfg.tight_profit_dte:
            if pnl_pct >= self.cfg.tight_profit_pct:
                return TradeAction.CLOSE_PROFIT
        else:
            if pnl_pct >= self.cfg.profit_target_pct:
                return TradeAction.CLOSE_PROFIT

        # 3. Stop loss
        if position.entry_credit > 0:
            current_cost = position.current_value
            if current_cost >= position.entry_credit * self.cfg.stop_loss_multiple:
                return TradeAction.CLOSE_STOP

        # 4. Roll check: if short strike is ITM near expiry
        if remaining_dte <= self.cfg.roll_dte_threshold:
            if spx_price < position.short_leg.strike:
                # Check roll eligibility
                if vix < 30 and spx_price > position.long_leg.strike:
                    return TradeAction.ROLL

        return TradeAction.HOLD

    def mark_to_market(
        self,
        position: SpreadPosition,
        spx_price: float,
        iv: float,
        as_of: Optional[date] = None,
        risk_free_rate: float = 0.05,
    ) -> SpreadPosition:
        """Update position's current value and greeks.

        Parameters
        ----------
        position : Position to update
        spx_price : Current SPX price
        iv : Current implied volatility
        as_of : Current date

        Returns
        -------
        Updated position (mutated in place and returned)
        """
        T = years_to_expiry(position.short_leg.expiry, as_of)

        if T <= 0:
            # At/past expiry — calculate intrinsic value
            short_intrinsic = max(position.short_leg.strike - spx_price, 0) * 100
            long_intrinsic = max(position.long_leg.strike - spx_price, 0) * 100
            position.current_value = short_intrinsic - long_intrinsic
        else:
            short_price = bs_put_price(
                spx_price, position.short_leg.strike, T, risk_free_rate, iv
            ) * 100
            long_price = bs_put_price(
                spx_price, position.long_leg.strike, T, risk_free_rate, iv
            ) * 100
            position.current_value = short_price - long_price

            # Update greeks
            position.short_leg.greeks = bs_greeks(
                spx_price, position.short_leg.strike, T, risk_free_rate, iv, "put"
            )
            position.long_leg.greeks = bs_greeks(
                spx_price, position.long_leg.strike, T, risk_free_rate, iv, "put"
            )

        return position


# ---------------------------------------------------------------------------
# VRP Strategy (Top-level)
# ---------------------------------------------------------------------------

class VRPStrategy:
    """Top-level strategy coordinator.

    Combines regime classification, strike selection, and position
    management into a single interface. The orchestrator calls:

    1. should_open_new_trade() — daily check for new entry
    2. evaluate_positions() — manage all open positions
    3. construct_spread() — build the actual spread to trade
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.regime = VIXRegimeClassifier(config.vix)
        self.selector = StrikeSelector(config.spread)
        self.manager = PositionManager(config.spread)
        self.positions: List[SpreadPosition] = []
        self._next_id = 0

    def should_open_new_trade(
        self,
        spx_price: float,
        vix: float,
        spx_200sma: Optional[float] = None,
        as_of: Optional[date] = None,
        signal_state: Optional[SignalState] = None,
    ) -> bool:
        """Determine if we should open a new spread today.

        Checks:
        - Signal layer allows trading (VRP, term structure, gap risk, events)
        - VIX regime allows trading
        - Not at max concurrent positions
        - SPX above 200-day SMA (trend filter, if enabled)
        - Total risk budget not exceeded
        """
        # Signal layer filter (if available)
        if signal_state is not None and not signal_state.can_trade:
            logger.info(f"No trade: signal blocked — {signal_state.reject_reason}")
            return False

        # VIX regime filter
        regime = self.regime.classify(vix)
        if regime in (VIXRegime.TOO_LOW, VIXRegime.CRISIS):
            logger.info(f"No trade: VIX regime = {regime.value} (VIX={vix:.1f})")
            return False

        # Position count limit
        open_positions = [p for p in self.positions if p.status == "open"]
        if len(open_positions) >= self.config.spread.max_concurrent_positions:
            logger.info(
                f"No trade: at max positions ({len(open_positions)}/{self.config.spread.max_concurrent_positions})"
            )
            return False

        # Trend filter
        if self.config.vix.require_uptrend and spx_200sma is not None:
            if spx_price < spx_200sma:
                logger.info(
                    f"No trade: SPX {spx_price:.0f} below 200-SMA {spx_200sma:.0f}"
                )
                return False

        # Risk budget check
        total_risk = sum(p.total_max_risk for p in open_positions)
        # We'll check against account value in the orchestrator
        return True

    def construct_spread(
        self,
        spx_price: float,
        vix: float,
        account_equity: float,
        expiry: Optional[date] = None,
        as_of: Optional[date] = None,
        risk_free_rate: float = 0.05,
        available_strikes: Optional[List[float]] = None,
        signal_state: Optional[SignalState] = None,
    ) -> Optional[SpreadPosition]:
        """Construct a new spread trade.

        Parameters
        ----------
        spx_price : Current SPX price
        vix : Current VIX level
        account_equity : Current account equity for sizing
        expiry : Target expiry (auto-selected if None)
        as_of : Current date
        risk_free_rate : Risk-free rate
        available_strikes : Real strikes from IBKR chain

        Returns
        -------
        A new SpreadPosition ready to execute, or None.
        """
        today = as_of or date.today()

        # Select expiry if not provided — find best expiry in DTE window
        if expiry is None:
            valid_expiries = self.selector.find_available_expiries(
                spx_price=spx_price, as_of=today
            )
            if not valid_expiries:
                # Fallback: find closest expiry to target DTE
                target = today + timedelta(days=self.config.spread.target_dte)
                exp = next_monthly_expiry(today)
                best = exp
                for _ in range(4):
                    if exp.month == 12:
                        exp = next_monthly_expiry(date(exp.year + 1, 1, 1))
                    else:
                        exp = next_monthly_expiry(date(exp.year, exp.month + 1, 1))
                    if abs((exp - target).days) < abs((best - target).days):
                        best = exp
                expiry = best
            else:
                # Choose the expiry closest to target DTE
                target_dte = self.config.spread.target_dte
                expiry = min(valid_expiries, key=lambda e: abs((e - today).days - target_dte))

        remaining = dte(expiry, today)
        if remaining < 7:  # absolute minimum
            logger.debug(f"No suitable expiry: {remaining} DTE too short")
            return None

        # Use VIX as IV proxy for theoretical pricing
        iv = vix / 100.0

        # Propagate account equity to StrikeSelector for dynamic width sizing
        self.selector._account_equity = account_equity

        # Build the spread
        result = self.selector.build_spread(
            spx_price=spx_price,
            expiry=expiry,
            iv=iv,
            vix=vix,
            as_of=today,
            risk_free_rate=risk_free_rate,
            available_strikes=available_strikes,
        )
        if result is None:
            return None

        short_leg, long_leg, net_credit = result

        # Position sizing — use ACTUAL spread width, not config default
        actual_width = short_leg.strike - long_leg.strike
        sizing_mult = self.regime.sizing_multiplier(vix)
        base_risk = actual_width * 100  # max loss per contract in dollars

        # Apply signal-layer sizing scalar (vol targeting, Kelly, gap risk)
        signal_scalar = 1.0
        if signal_state is not None:
            signal_scalar = signal_state.sizing_scalar

        risk_budget = account_equity * self.config.spread.risk_per_trade * sizing_mult * signal_scalar

        quantity = max(1, int(risk_budget / base_risk))

        # Check total risk budget
        open_risk = sum(
            p.total_max_risk for p in self.positions if p.status == "open"
        )
        max_total = account_equity * self.config.spread.max_total_risk_pct
        remaining_budget = max_total - open_risk
        if remaining_budget < base_risk:
            logger.info("No trade: total risk budget exhausted")
            return None

        quantity = min(quantity, int(remaining_budget / base_risk))
        if quantity <= 0:
            return None

        # Create position
        self._next_id += 1
        position = SpreadPosition(
            id=f"VRP-{self._next_id:04d}",
            short_leg=short_leg,
            long_leg=long_leg,
            entry_date=today,
            entry_credit=net_credit,
            quantity=quantity,
            current_value=net_credit,  # at entry, current = credit
            spx_at_entry=spx_price,
            vix_at_entry=vix,
        )

        self.positions.append(position)
        logger.info(
            f"New spread {position.id}: "
            f"sell {short_leg.strike}P / buy {long_leg.strike}P "
            f"exp {expiry} | credit ${net_credit:.0f} x {quantity} | "
            f"max risk ${position.total_max_risk:,.0f} | "
            f"VIX={vix:.1f} SPX={spx_price:.0f}"
        )

        return position

    def evaluate_positions(
        self,
        spx_price: float,
        vix: float,
        iv: Optional[float] = None,
        as_of: Optional[date] = None,
        risk_free_rate: float = 0.05,
        spx_low: Optional[float] = None,
    ) -> List[Tuple[SpreadPosition, TradeAction]]:
        """Evaluate all open positions and return recommended actions.

        Parameters
        ----------
        spx_price : Current SPX price
        vix : Current VIX level
        iv : Implied volatility for pricing (defaults to VIX/100)
        as_of : Current date
        risk_free_rate : Risk-free rate for BS pricing
        spx_low : Intraday SPX low for short-strike breach detection

        Returns
        -------
        List of (position, action) tuples for positions needing action.
        """
        if iv is None:
            iv = vix / 100.0

        actions = []
        for position in self.positions:
            if position.status != "open":
                continue

            # Mark to market
            self.manager.mark_to_market(
                position, spx_price, iv, as_of, risk_free_rate
            )

            # Evaluate — pass spx_low for intraday breach detection
            action = self.manager.evaluate(
                position, spx_price, vix, as_of, spx_low=spx_low
            )

            if action != TradeAction.HOLD:
                actions.append((position, action))

        return actions

    def close_position(
        self,
        position: SpreadPosition,
        reason: str,
        close_value: Optional[float] = None,
        as_of: Optional[date] = None,
    ) -> float:
        """Close a position and return realized P&L.

        Parameters
        ----------
        position : Position to close
        reason : Why we're closing
        close_value : Cost to close (use current_value if None)
        as_of : Close date

        Returns
        -------
        Realized P&L in dollars
        """
        today = as_of or date.today()

        if close_value is None:
            close_value = position.current_value

        pnl = (position.entry_credit - close_value) * position.quantity
        position.status = "closed"
        position.close_date = today
        position.close_pnl = pnl
        position.close_reason = reason

        logger.info(
            f"Closed {position.id}: {reason} | "
            f"P&L ${pnl:+,.0f} ({position.pnl_pct:+.1%}) | "
            f"held {position.days_held}d"
        )

        return pnl

    @property
    def open_positions(self) -> List[SpreadPosition]:
        """Return all currently open positions."""
        return [p for p in self.positions if p.status == "open"]

    @property
    def portfolio_greeks(self) -> Dict[str, float]:
        """Aggregate portfolio-level greeks."""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for pos in self.open_positions:
            qty = pos.quantity
            if pos.short_leg.greeks:
                g = pos.short_leg.greeks
                # Short leg: negate because we sold
                total_delta -= g.delta * qty * 100
                total_gamma -= g.gamma * qty * 100
                total_theta -= g.theta * qty * 100
                total_vega -= g.vega * qty * 100
            if pos.long_leg.greeks:
                g = pos.long_leg.greeks
                # Long leg: add because we bought
                total_delta += g.delta * qty * 100
                total_gamma += g.gamma * qty * 100
                total_theta += g.theta * qty * 100
                total_vega += g.vega * qty * 100

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta": total_theta,
            "vega": total_vega,
        }
