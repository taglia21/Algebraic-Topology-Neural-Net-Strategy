"""
Position Sizing — Fixed Fractional
====================================

Simple, robust position sizing that risks a fixed 1 % of portfolio per trade.

Why NOT Kelly:
  Kelly requires accurate probability-of-profit estimates.  Our PoP is a
  heuristic (rule-based, not calibrated), so Kelly amplifies estimation error
  and produces dangerously large sizes when PoP is overestimated.

Formula:
  max_risk       = portfolio_value × 0.01          (1 % of equity)
  contracts      = floor(max_risk / max_loss_per_contract)
  contracts      = clamp(contracts, 1, max_contracts_per_trade)  (cap = 3)

The module still exposes the same ``MedallionPositionSizer`` /
``PositionSize`` interface so the rest of the engine is unchanged.
"""

import logging
import math
from typing import Any, Dict, Optional
from dataclasses import dataclass

from .config import RISK_CONFIG, VOLATILITY_REGIMES


def _get_position_sizer_config() -> Dict[str, Any]:
    config: Dict[str, Any] = dict(RISK_CONFIG)
    config.setdefault("volatility_regimes", VOLATILITY_REGIMES)
    config.setdefault(
        "max_contracts_per_symbol",
        config.get("max_contracts_per_trade", 3),
    )
    return config


# Fixed fraction of portfolio risked per trade
FIXED_RISK_FRACTION = 0.01  # 1 %


@dataclass
class PositionSize:
    """Position size recommendation."""
    contracts: int
    dollar_amount: float
    risk_dollar_amount: float
    risk_percent: float
    kelly_fraction: float          # kept for interface compat — always 0.0
    confidence_multiplier: float
    volatility_multiplier: float
    reason: str


class MedallionPositionSizer:
    """
    Fixed-fractional position sizer.

    Risks 1 % of portfolio per trade, capped at ``max_contracts_per_trade``.
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
    ) -> PositionSize:
        """
        Calculate position size using fixed-fractional method.

        contracts = floor(portfolio_value * 1% / max_loss_per_contract)
        Capped at max_contracts_per_trade (default 3).
        """
        max_risk = portfolio_value * FIXED_RISK_FRACTION

        # Calculate raw contracts
        if max_loss_per_contract <= 0:
            self.logger.warning("Invalid max_loss_per_contract, using minimum position")
            contracts = 1
        else:
            contracts = math.floor(max_risk / max_loss_per_contract)

        # Clamp to [1, max_contracts_per_trade]
        contracts = max(1, contracts)
        contracts = min(contracts, self.config["max_contracts_per_symbol"])

        # ===== 2026-02-23 FIX 7: Cap at 2% of portfolio =====
        max_single_pct = self.config.get("max_single_position_pct", 0.02)
        max_pos_value = portfolio_value * max_single_pct
        if max_loss_per_contract > 0:
            max_by_value = max(1, int(max_pos_value / max_loss_per_contract))
            if contracts > max_by_value:
                contracts = max_by_value

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

        reason = (
            f"FixedFrac 1%: ${max_risk:,.0f} budget, "
            f"{contracts} contracts, "
            f"Risk: {risk_pct:.2%}"
        )

        return PositionSize(
            contracts=contracts,
            dollar_amount=risk_dollar,
            risk_dollar_amount=risk_dollar,
            risk_percent=risk_pct,
            kelly_fraction=0.0,
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
