"""
Delta Hedger — Automated Delta-Neutral Hedge Execution
=======================================================

Calculates hedge quantities and executes trades to bring portfolio
delta within acceptable bands.  Supports hedging via:
  • Underlying shares (primary — cheapest, most direct)
  • Options (puts/calls for delta + convexity adjustment)

Configurable delta bands ensure the portfolio stays delta-neutral
within tolerance without excessive rebalancing (transaction cost aware).

Usage:
    from src.delta_hedger import DeltaHedger, HedgeConfig

    hedger = DeltaHedger()
    rec = hedger.get_hedge_recommendation(portfolio_delta=-150, underlying="SPY", spot=495.0)
    if rec.should_hedge:
        hedger.execute_delta_hedge(rec)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HedgeConfig:
    """Configuration for delta hedging behaviour."""
    # Delta bands
    delta_hedge_threshold: float = 0.30        # Hedge when |portfolio delta| > this (per $100k)
    max_portfolio_delta: float = 0.50           # Hard cap — must hedge immediately
    target_delta_after_hedge: float = 0.0       # Target post-hedge delta (0 = fully neutral)

    # Hedge instrument preference
    prefer_shares: bool = True                  # True = hedge with shares, False = options
    min_shares_hedge: int = 10                  # Don't hedge fewer than 10 shares (tx cost)
    max_hedge_notional_pct: float = 0.10        # Max 10% of equity for a single hedge

    # Options hedge params (when prefer_shares=False)
    hedge_option_dte: int = 30                  # DTE for hedge options
    hedge_option_delta_target: float = 0.5      # ATM options for hedging

    # Transaction cost awareness
    min_delta_change: float = 5.0               # Don't hedge if change < 5 deltas
    rebalance_cooldown_sec: int = 300           # Min 5 min between hedges

    # Paper trading safety
    paper_trading: bool = True


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class HedgeInstrument(Enum):
    SHARES = "shares"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"


class HedgeUrgency(Enum):
    NONE = "none"               # Within tolerance
    LOW = "low"                 # Approaching threshold
    MEDIUM = "medium"           # Exceeded threshold
    HIGH = "high"               # Exceeded hard cap
    CRITICAL = "critical"       # Circuit breaker territory


@dataclass
class HedgeRecommendation:
    """Recommendation for a delta hedge trade."""
    should_hedge: bool = False
    urgency: HedgeUrgency = HedgeUrgency.NONE
    instrument: HedgeInstrument = HedgeInstrument.SHARES
    underlying: str = ""
    direction: str = ""          # "buy" or "sell"
    quantity: int = 0            # Shares or contracts
    estimated_cost: float = 0.0  # Approximate notional
    current_delta: float = 0.0
    target_delta: float = 0.0
    delta_reduction: float = 0.0
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # For options hedges
    option_type: str = ""        # "call" or "put"
    option_strike: float = 0.0
    option_dte: int = 0

    def to_dict(self) -> dict:
        return {
            "should_hedge": self.should_hedge,
            "urgency": self.urgency.value,
            "instrument": self.instrument.value,
            "underlying": self.underlying,
            "direction": self.direction,
            "quantity": self.quantity,
            "estimated_cost": round(self.estimated_cost, 2),
            "current_delta": round(self.current_delta, 2),
            "target_delta": round(self.target_delta, 2),
            "delta_reduction": round(self.delta_reduction, 2),
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HedgeExecution:
    """Record of an executed hedge trade."""
    recommendation: HedgeRecommendation
    executed: bool = False
    execution_time: datetime = field(default_factory=datetime.now)
    order_id: str = ""
    fill_price: float = 0.0
    fill_qty: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "executed": self.executed,
            "execution_time": self.execution_time.isoformat(),
            "order_id": self.order_id,
            "fill_price": self.fill_price,
            "fill_qty": self.fill_qty,
            "error": self.error,
            **self.recommendation.to_dict(),
        }


# ============================================================================
# DELTA HEDGER
# ============================================================================

class DeltaHedger:
    """
    Calculates and executes delta hedges for the options portfolio.

    The hedger computes the number of shares (or option contracts) needed
    to bring portfolio delta back to the target level, then executes via
    the provided trading client.
    """

    def __init__(
        self,
        config: Optional[HedgeConfig] = None,
        trading_client=None,
        options_engine=None,
    ):
        """
        Args:
            config: HedgeConfig with band parameters
            trading_client: Alpaca TradingClient (or None for dry-run)
            options_engine: AlpacaOptionsEngine (for options hedges)
        """
        self.config = config or HedgeConfig()
        self._trading_client = trading_client
        self._options_engine = options_engine
        self._last_hedge_time: Optional[datetime] = None
        self._hedge_history: List[HedgeExecution] = []
        self._max_history = 200
        logger.info("DeltaHedger initialized")

    # ── Core: calculate hedge quantity ──────────────────────────────

    def calculate_hedge_quantity(
        self,
        portfolio_delta: float,
        underlying: str = "SPY",
        spot_price: float = 0.0,
        equity: float = 100_000.0,
    ) -> Tuple[int, str]:
        """
        Calculate the number of shares needed to hedge portfolio delta.

        For shares, each share contributes delta=1.0, so:
            hedge_shares = -portfolio_delta  (sell if delta > 0, buy if delta < 0)

        Args:
            portfolio_delta: Current net portfolio delta
            underlying: Underlying to hedge with
            spot_price: Current price of underlying
            equity: Account equity ($)

        Returns:
            (shares_to_trade, direction)
            Positive qty, direction = "buy" or "sell"
        """
        target = self.config.target_delta_after_hedge
        delta_to_offset = portfolio_delta - target

        if abs(delta_to_offset) < self.config.min_delta_change:
            return 0, ""

        # For shares: 1 share = 1 delta
        raw_shares = int(round(-delta_to_offset))

        if abs(raw_shares) < self.config.min_shares_hedge:
            return 0, ""

        # Cap by max hedge notional
        if spot_price > 0:
            max_shares = int(equity * self.config.max_hedge_notional_pct / spot_price)
            raw_shares = max(-max_shares, min(max_shares, raw_shares))

        direction = "buy" if raw_shares > 0 else "sell"
        qty = abs(raw_shares)

        return qty, direction

    # ── Core: get recommendation ────────────────────────────────────

    def get_hedge_recommendation(
        self,
        portfolio_delta: float,
        underlying: str = "SPY",
        spot_price: float = 0.0,
        equity: float = 100_000.0,
        portfolio_gamma: float = 0.0,
        portfolio_vega: float = 0.0,
    ) -> HedgeRecommendation:
        """
        Generate a hedge recommendation based on current portfolio Greeks.

        Args:
            portfolio_delta: Net portfolio delta
            underlying: Primary hedging instrument
            spot_price: Current underlying price
            equity: Account equity
            portfolio_gamma: Net gamma (for urgency assessment)
            portfolio_vega: Net vega (for urgency assessment)

        Returns:
            HedgeRecommendation with action details
        """
        rec = HedgeRecommendation(
            underlying=underlying,
            current_delta=portfolio_delta,
            target_delta=self.config.target_delta_after_hedge,
            timestamp=datetime.now(),
        )

        abs_delta = abs(portfolio_delta)

        # Determine urgency
        if abs_delta <= self.config.delta_hedge_threshold * 0.5:
            rec.urgency = HedgeUrgency.NONE
            rec.reason = f"Delta {portfolio_delta:+.1f} within safe band"
            return rec
        elif abs_delta <= self.config.delta_hedge_threshold:
            rec.urgency = HedgeUrgency.LOW
            rec.reason = f"Delta {portfolio_delta:+.1f} approaching threshold"
            return rec  # Don't hedge yet, just warn
        elif abs_delta <= self.config.max_portfolio_delta:
            rec.urgency = HedgeUrgency.MEDIUM
        elif abs_delta <= self.config.max_portfolio_delta * 1.5:
            rec.urgency = HedgeUrgency.HIGH
        else:
            rec.urgency = HedgeUrgency.CRITICAL

        # Check cooldown
        if self._last_hedge_time:
            elapsed = (datetime.now() - self._last_hedge_time).total_seconds()
            if elapsed < self.config.rebalance_cooldown_sec and rec.urgency != HedgeUrgency.CRITICAL:
                rec.should_hedge = False
                rec.reason = f"Cooldown active ({int(elapsed)}s / {self.config.rebalance_cooldown_sec}s)"
                return rec

        # Calculate hedge
        if self.config.prefer_shares:
            qty, direction = self.calculate_hedge_quantity(
                portfolio_delta, underlying, spot_price, equity
            )
            if qty > 0:
                rec.should_hedge = True
                rec.instrument = HedgeInstrument.SHARES
                rec.direction = direction
                rec.quantity = qty
                rec.estimated_cost = qty * spot_price if spot_price > 0 else 0.0
                rec.delta_reduction = portfolio_delta - self.config.target_delta_after_hedge
                rec.reason = (
                    f"Delta {portfolio_delta:+.1f} exceeds threshold "
                    f"{self.config.delta_hedge_threshold:.1f} — "
                    f"{direction.upper()} {qty} shares of {underlying}"
                )
            else:
                rec.should_hedge = False
                rec.reason = f"Delta change too small to hedge ({abs(portfolio_delta):.1f})"
        else:
            # Options-based hedge
            rec = self._build_options_hedge_rec(
                rec, portfolio_delta, underlying, spot_price, equity
            )

        return rec

    # ── Core: execute hedge ─────────────────────────────────────────

    def execute_delta_hedge(
        self,
        recommendation: HedgeRecommendation,
        dry_run: bool = False,
    ) -> HedgeExecution:
        """
        Execute a hedge trade based on recommendation.

        Args:
            recommendation: HedgeRecommendation from get_hedge_recommendation()
            dry_run: If True, log but don't execute

        Returns:
            HedgeExecution with result details
        """
        execution = HedgeExecution(
            recommendation=recommendation,
            execution_time=datetime.now(),
        )

        if not recommendation.should_hedge:
            execution.error = "No hedge needed"
            return execution

        if dry_run or self.config.paper_trading:
            # Simulate execution
            execution.executed = True
            execution.fill_qty = recommendation.quantity
            execution.fill_price = recommendation.estimated_cost / max(1, recommendation.quantity)
            execution.order_id = f"DRY-{datetime.now().strftime('%H%M%S')}"
            self._last_hedge_time = datetime.now()
            logger.info(
                f"🔄 HEDGE (dry-run): {recommendation.direction.upper()} "
                f"{recommendation.quantity} {recommendation.instrument.value} "
                f"{recommendation.underlying} — delta {recommendation.current_delta:+.1f} "
                f"→ {recommendation.target_delta:+.1f}"
            )
            self._record_execution(execution)
            return execution

        # Real execution via Alpaca
        if recommendation.instrument == HedgeInstrument.SHARES:
            execution = self._execute_shares_hedge(recommendation, execution)
        elif recommendation.instrument in (HedgeInstrument.CALL_OPTION, HedgeInstrument.PUT_OPTION):
            execution = self._execute_options_hedge(recommendation, execution)
        else:
            execution.error = f"Unknown instrument: {recommendation.instrument}"

        self._record_execution(execution)
        return execution

    # ── Execution helpers ───────────────────────────────────────────

    def _execute_shares_hedge(
        self,
        rec: HedgeRecommendation,
        execution: HedgeExecution,
    ) -> HedgeExecution:
        """Execute a share-based hedge via Alpaca TradingClient."""
        if self._trading_client is None:
            execution.error = "No trading client available"
            return execution

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side = OrderSide.BUY if rec.direction == "buy" else OrderSide.SELL
            order_request = MarketOrderRequest(
                symbol=rec.underlying,
                qty=rec.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
            )

            order = self._trading_client.submit_order(order_request)
            execution.executed = True
            execution.order_id = str(getattr(order, "id", ""))
            execution.fill_qty = rec.quantity
            execution.fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
            self._last_hedge_time = datetime.now()

            logger.info(
                f"✅ HEDGE EXECUTED: {rec.direction.upper()} "
                f"{rec.quantity} shares {rec.underlying} "
                f"(order {execution.order_id})"
            )

        except Exception as e:
            execution.error = str(e)
            logger.error(f"Hedge execution failed: {e}")

        return execution

    def _execute_options_hedge(
        self,
        rec: HedgeRecommendation,
        execution: HedgeExecution,
    ) -> HedgeExecution:
        """Execute an options-based hedge."""
        if self._options_engine is None:
            execution.error = "No options engine available"
            return execution

        try:
            side = "buy" if rec.direction == "buy" else "sell"
            result = self._options_engine.place_option_order(
                symbol=rec.option_type,  # Would need resolved contract symbol
                quantity=rec.quantity,
                side=side,
            )
            if result:
                execution.executed = True
                execution.order_id = str(result.get("id", ""))
                execution.fill_qty = rec.quantity
                self._last_hedge_time = datetime.now()
                logger.info(
                    f"✅ HEDGE OPTION: {rec.direction.upper()} "
                    f"{rec.quantity} {rec.option_type} contracts"
                )
            else:
                execution.error = "Options order returned None"

        except Exception as e:
            execution.error = str(e)
            logger.error(f"Options hedge execution failed: {e}")

        return execution

    # ── Options hedge recommendation builder ────────────────────────

    def _build_options_hedge_rec(
        self,
        rec: HedgeRecommendation,
        portfolio_delta: float,
        underlying: str,
        spot_price: float,
        equity: float,
    ) -> HedgeRecommendation:
        """Build a hedge recommendation using options instead of shares."""
        target_delta = self.config.target_delta_after_hedge
        delta_to_offset = portfolio_delta - target_delta

        if abs(delta_to_offset) < self.config.min_delta_change:
            rec.should_hedge = False
            rec.reason = "Delta change too small"
            return rec

        # Use ATM option with ~0.5 delta per contract = 50 delta exposure per contract
        option_delta_per_contract = 50  # Approximate: 0.50 delta x 100 shares
        contracts_needed = int(round(abs(delta_to_offset) / option_delta_per_contract))
        contracts_needed = max(1, contracts_needed)

        if delta_to_offset > 0:
            # Portfolio is too long → buy puts or sell calls
            rec.instrument = HedgeInstrument.PUT_OPTION
            rec.option_type = "put"
            rec.direction = "buy"
        else:
            # Portfolio is too short → buy calls or sell puts
            rec.instrument = HedgeInstrument.CALL_OPTION
            rec.option_type = "call"
            rec.direction = "buy"

        rec.should_hedge = True
        rec.quantity = contracts_needed
        rec.option_strike = round(spot_price, 0) if spot_price > 0 else 0
        rec.option_dte = self.config.hedge_option_dte
        rec.delta_reduction = delta_to_offset
        rec.estimated_cost = contracts_needed * spot_price * 0.02  # ~2% of notional approx
        rec.reason = (
            f"Delta {portfolio_delta:+.1f} exceeds threshold — "
            f"{rec.direction.upper()} {contracts_needed} ATM {rec.option_type}s "
            f"on {underlying} (DTE={rec.option_dte})"
        )

        return rec

    # ── History & state ─────────────────────────────────────────────

    def _record_execution(self, execution: HedgeExecution):
        self._hedge_history.append(execution)
        if len(self._hedge_history) > self._max_history:
            self._hedge_history = self._hedge_history[-self._max_history:]

    def get_hedge_history(self) -> List[HedgeExecution]:
        return list(self._hedge_history)

    def get_last_hedge_time(self) -> Optional[datetime]:
        return self._last_hedge_time

    def get_hedge_stats(self) -> dict:
        """Summary statistics for executed hedges."""
        executed = [h for h in self._hedge_history if h.executed]
        return {
            "total_hedges": len(executed),
            "total_attempted": len(self._hedge_history),
            "share_hedges": sum(
                1 for h in executed
                if h.recommendation.instrument == HedgeInstrument.SHARES
            ),
            "option_hedges": sum(
                1 for h in executed
                if h.recommendation.instrument != HedgeInstrument.SHARES
            ),
            "last_hedge_time": (
                self._last_hedge_time.isoformat() if self._last_hedge_time else None
            ),
        }


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    hedger = DeltaHedger()

    # Test 1: Small delta — no hedge
    rec1 = hedger.get_hedge_recommendation(
        portfolio_delta=10.0, underlying="SPY", spot_price=495.0
    )
    print(f"Small delta: should_hedge={rec1.should_hedge}, urgency={rec1.urgency.value}")
    assert not rec1.should_hedge

    # Test 2: Medium delta — hedge with shares
    rec2 = hedger.get_hedge_recommendation(
        portfolio_delta=150.0, underlying="SPY", spot_price=495.0
    )
    print(f"Large delta: should_hedge={rec2.should_hedge}, "
          f"qty={rec2.quantity}, dir={rec2.direction}, "
          f"urgency={rec2.urgency.value}")
    assert rec2.should_hedge
    assert rec2.direction == "sell"  # Positive delta → sell shares to offset

    # Test 3: Negative delta — buy shares
    rec3 = hedger.get_hedge_recommendation(
        portfolio_delta=-200.0, underlying="SPY", spot_price=495.0
    )
    print(f"Negative delta: should_hedge={rec3.should_hedge}, "
          f"qty={rec3.quantity}, dir={rec3.direction}")
    assert rec3.should_hedge
    assert rec3.direction == "buy"

    # Test 4: Execute dry-run hedge
    exec_result = hedger.execute_delta_hedge(rec2, dry_run=True)
    print(f"Execution: executed={exec_result.executed}, order={exec_result.order_id}")
    assert exec_result.executed

    # Test 5: Stats
    stats = hedger.get_hedge_stats()
    print(f"Stats: {stats}")
    assert stats["total_hedges"] == 1

    print("Delta hedger OK")
