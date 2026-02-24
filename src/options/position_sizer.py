"""
Position Sizing — Kelly-Capped with Black-Scholes PoP
======================================================

Sizes positions using the Kelly criterion with a properly calibrated
probability-of-profit derived from Black-Scholes N(d2) — the risk-neutral
probability that a short option expires worthless.

Key rules:
  1. PoP = N(d2)  for short options (credit spreads, iron condors, etc.)
  2. Kelly fraction = PoP - (1-PoP) / (win/loss ratio)
  3. Fraction CAPPED at min(kelly, 0.15) to avoid over-sizing
  4. Max single position = 1% of portfolio value (hard $700 cap on $70k acct)
  5. Delta constraint still enforced

The module exposes ``MedallionPositionSizer`` / ``PositionSize`` for
drop-in compatibility with the rest of the engine.
"""

import logging
import math
from typing import Any, Dict, Optional
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .config import RISK_CONFIG, VOLATILITY_REGIMES


def _get_position_sizer_config() -> Dict[str, Any]:
    config: Dict[str, Any] = dict(RISK_CONFIG)
    config.setdefault("volatility_regimes", VOLATILITY_REGIMES)
    config.setdefault(
        "max_contracts_per_symbol",
        config.get("max_contracts_per_trade", 3),
    )
    return config


# Hard limits
MAX_KELLY_FRACTION = 0.15          # Cap fractional Kelly
MAX_SINGLE_POSITION_PCT = 0.01    # 1% of portfolio
FIXED_RISK_FRACTION = 0.01        # Fallback if Kelly unavailable
RISK_FREE_RATE = 0.05             # ~5% for money-market rate


def _bs_d2(
    S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE
) -> float:
    """Black-Scholes d2 parameter.

    d2 = [ln(S/K) + (r - 0.5*sigma^2)*T] / (sigma*sqrt(T))
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d2


def black_scholes_pop(
    underlying_price: float,
    strike: float,
    dte: float,
    iv: float,
    option_type: str = "put",
) -> float:
    """Probability that a SHORT option expires worthless (OTM).

    For a short put:  PoP = N(d2)   — probability S > K at expiration
    For a short call: PoP = N(-d2)  — probability S < K at expiration

    Args:
        underlying_price: Current underlying price.
        strike: Strike price of the short option.
        dte: Days to expiration.
        iv: Annualised implied volatility (e.g. 0.25 for 25%).
        option_type: ``"put"`` or ``"call"``.

    Returns:
        Probability of profit in [0, 1].
    """
    T = max(dte, 1) / 365.0
    d2 = _bs_d2(underlying_price, strike, T, iv)

    if option_type.lower() == "put":
        return float(norm.cdf(d2))   # P(S > K)
    else:
        return float(norm.cdf(-d2))  # P(S < K)


def kelly_fraction(
    pop: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Compute Kelly fraction, capped at MAX_KELLY_FRACTION.

    Kelly = PoP - (1 - PoP) / (avg_win / avg_loss)

    A negative Kelly means the trade has negative expected value;
    returns 0.0 in that case.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    win_loss_ratio = avg_win / avg_loss
    k = pop - (1 - pop) / win_loss_ratio
    if k <= 0:
        return 0.0
    return min(k, MAX_KELLY_FRACTION)


@dataclass
class PositionSize:
    """Position size recommendation."""
    contracts: int
    dollar_amount: float
    risk_dollar_amount: float
    risk_percent: float
    kelly_fraction: float          # actual Kelly value used (0.0–0.15)
    confidence_multiplier: float
    volatility_multiplier: float
    reason: str


class MedallionPositionSizer:
    """
    Kelly-capped position sizer with Black-Scholes PoP.

    Calculates fractional Kelly when option pricing data is available;
    falls back to fixed 1% risk fraction otherwise.  Hard caps:
      - Kelly fraction ≤ 0.15
      - Single position ≤ 1% of portfolio
      - Contracts ≤ max_contracts_per_trade
    """

    def __init__(self):
        self.config = _get_position_sizer_config()
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(
        self,
        portfolio_value: float,
        max_loss_per_contract: float,
        signal_confidence: float,
        probability_of_profit: Optional[float] = None,
        iv_rank: Optional[float] = None,
        current_portfolio_delta: float = 0.0,
        position_delta_per_contract: float = 0.0,
        # New BS-PoP params (optional — caller can provide for Kelly)
        underlying_price: Optional[float] = None,
        strike: Optional[float] = None,
        dte: Optional[float] = None,
        iv: Optional[float] = None,
        option_type: str = "put",
        expected_premium: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size using Kelly when BS inputs are available,
        else fall back to fixed-fractional 1%.

        Hard caps:
          - Kelly ≤ 0.15
          - Risk ≤ 1% of portfolio
          - Contracts ≤ max_contracts_per_trade
        """
        # --- Try Kelly via Black-Scholes PoP ---
        used_kelly = 0.0
        if (
            underlying_price is not None
            and strike is not None
            and dte is not None
            and iv is not None
            and iv > 0
            and max_loss_per_contract > 0
        ):
            pop = black_scholes_pop(underlying_price, strike, dte, iv, option_type)
            avg_win = (expected_premium or 0) * 100  # premium collected per contract
            avg_loss = max_loss_per_contract
            if avg_win > 0:
                used_kelly = kelly_fraction(pop, avg_win, avg_loss)
                self.logger.debug(
                    f"Kelly={used_kelly:.3f} (PoP={pop:.2%}, "
                    f"win=${avg_win:.0f}, loss=${avg_loss:.0f})"
                )

        # --- Compute risk budget ---
        if used_kelly > 0:
            risk_budget = portfolio_value * used_kelly
        else:
            risk_budget = portfolio_value * FIXED_RISK_FRACTION

        # --- Hard cap: 1% of portfolio ---
        max_pos_value = portfolio_value * MAX_SINGLE_POSITION_PCT
        risk_budget = min(risk_budget, max_pos_value)

        # --- Contracts ---
        if max_loss_per_contract <= 0:
            self.logger.warning("Invalid max_loss_per_contract, using minimum position")
            contracts = 1
        else:
            contracts = math.floor(risk_budget / max_loss_per_contract)

        # Clamp to [1, max_contracts_per_trade]
        contracts = max(1, contracts)
        contracts = min(contracts, self.config["max_contracts_per_symbol"])

        # Delta constraint
        if position_delta_per_contract != 0:
            max_portfolio_delta = self.config["max_portfolio_delta"]
            new_delta = current_portfolio_delta + (contracts * position_delta_per_contract)
            if abs(new_delta) > max_portfolio_delta:
                headroom = max_portfolio_delta - abs(current_portfolio_delta)
                max_by_delta = int(headroom / abs(position_delta_per_contract))
                if max_by_delta < contracts:
                    contracts = max(0, max_by_delta)
                    self.logger.info(
                        f"Reduced to {contracts} contracts due to delta constraint"
                    )

        # Final bookkeeping
        risk_dollar = contracts * max_loss_per_contract
        risk_pct = risk_dollar / portfolio_value if portfolio_value > 0 else 0.0

        method = f"Kelly={used_kelly:.3f}" if used_kelly > 0 else "FixedFrac 1%"
        reason = (
            f"{method}: ${risk_budget:,.0f} budget, "
            f"{contracts} contracts, "
            f"Risk: {risk_pct:.2%}"
        )

        return PositionSize(
            contracts=contracts,
            dollar_amount=risk_dollar,
            risk_dollar_amount=risk_dollar,
            risk_percent=risk_pct,
            kelly_fraction=used_kelly,
            confidence_multiplier=signal_confidence,
            volatility_multiplier=1.0,
            reason=reason,
        )
    
    def validate_position_size(
        self,
        position_size: PositionSize,
        portfolio_value: float,
    ) -> bool:
        """
        Validate position size meets all constraints.
        
        Args:
            position_size: Calculated position size
            portfolio_value: Total portfolio value
            
        Returns:
            True if valid, False otherwise
        """
        # Check max risk per trade
        if position_size.risk_percent > self.config["max_risk_per_trade_pct"]:
            self.logger.warning(
                f"Position risk {position_size.risk_percent:.2%} exceeds "
                f"max {self.config['max_risk_per_trade_pct']:.2%}"
            )
            return False
        
        # Check max contracts
        if position_size.contracts > self.config["max_contracts_per_symbol"]:
            self.logger.warning(
                f"Contracts {position_size.contracts} exceeds "
                f"max {self.config['max_contracts_per_symbol']}"
            )
            return False
        
        # Check minimum contracts
        if position_size.contracts < 1:
            self.logger.warning("Position size less than 1 contract")
            return False
        
        return True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_max_loss_per_contract(
    strategy: str,
    strike_width: float = 5.0,
    premium_received: float = 0.0,
) -> float:
    """
    Calculate max loss per contract for common strategies.
    
    Args:
        strategy: Strategy name
        strike_width: Width between strikes ($)
        premium_received: Premium received per contract ($)
        
    Returns:
        Max loss per contract ($)
    """
    # Credit spreads: Max loss = (strike_width - premium) * 100
    if strategy in ["credit_spread", "put_spread", "iron_condor"]:
        return (strike_width - premium_received) * 100
    
    # Debit spreads: Max loss = premium paid * 100
    elif strategy in ["debit_spread", "call_spread"]:
        return premium_received * 100  # Premium is cost here
    
    # Straddles/Strangles: Max loss = premium * 100
    elif strategy in ["straddle", "strangle"]:
        return premium_received * 100
    
    # Default: Assume $500 max loss
    else:
        return 500.0


def estimate_position_delta(
    strategy: str,
    contracts: int,
    underlying_delta: float = 0.0,
) -> float:
    """
    Estimate position delta.
    
    Args:
        strategy: Strategy name
        contracts: Number of contracts
        underlying_delta: Delta of underlying option
        
    Returns:
        Estimated position delta
    """
    # Iron condor: Approximately delta neutral
    if strategy == "iron_condor":
        return 0.0 * contracts
    
    # Credit spread: Slightly negative delta
    elif strategy == "credit_spread":
        return -0.10 * contracts
    
    # Put spread: Positive delta (bullish)
    elif strategy == "put_spread":
        return 0.20 * contracts
    
    # Straddle: Near zero at ATM
    elif strategy == "straddle":
        return 0.0 * contracts
    
    # Use provided underlying delta
    else:
        return underlying_delta * contracts


# ============================================================================
# GRAND OVERHAUL: Portfolio Heat & Correlation Adjustments
# ============================================================================

def portfolio_heat_check(
    total_portfolio_vega: float,
    avg_daily_vega: float,
) -> bool:
    """Reject new position if total vega > 2x average daily vega.

    Args:
        total_portfolio_vega: Sum of abs(vega) across all positions.
        avg_daily_vega: Average daily vega move (e.g. from VIX history).

    Returns:
        True if portfolio heat is acceptable, False if new position should
        be rejected.
    """
    if avg_daily_vega <= 0:
        return True  # Cannot evaluate; allow
    ratio = total_portfolio_vega / avg_daily_vega
    if ratio > 2.0:
        logging.getLogger(__name__).warning(
            f"PORTFOLIO_HEAT: total vega {total_portfolio_vega:.2f} > "
            f"2x avg daily vega {avg_daily_vega:.2f} (ratio={ratio:.2f}) — REJECT"
        )
        return False
    return True


def correlation_adjustment(
    base_contracts: int,
    new_position_correlation: float,
    threshold: float = 0.7,
    reduction_pct: float = 0.30,
) -> int:
    """Reduce size 30% if new position correlation > 0.7 with book.

    Args:
        base_contracts: Original number of contracts.
        new_position_correlation: Pairwise correlation of new position
            with existing book (0 to 1).
        threshold: Correlation threshold to trigger reduction (default 0.7).
        reduction_pct: Fraction to reduce by (default 0.30 = 30%).

    Returns:
        Adjusted number of contracts (at least 1).
    """
    if new_position_correlation > threshold:
        reduced = int(base_contracts * (1.0 - reduction_pct))
        reduced = max(1, reduced)
        logging.getLogger(__name__).info(
            f"CORR_ADJ: corr={new_position_correlation:.2f} > {threshold} "
            f"→ reduced {base_contracts} → {reduced} contracts"
        )
        return reduced
    return base_contracts
