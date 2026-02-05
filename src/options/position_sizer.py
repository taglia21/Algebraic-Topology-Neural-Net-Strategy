"""
Position Sizing using Kelly Criterion
======================================

Implements fractional Kelly position sizing with risk adjustments.

Key features:
- Kelly Criterion: Optimal position size based on edge and odds
- Fractional Kelly: Conservative 0.25x Kelly for safety
- Volatility adjustment: Scale size based on IV rank regime
- Portfolio constraints: Max 2% risk per trade, max portfolio delta
- Signal confidence scaling: Higher confidence = larger size
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

from .config import RISK_CONFIG, VOLATILITY_REGIMES


def _get_position_sizer_config() -> Dict[str, Any]:
    config: Dict[str, Any] = dict(RISK_CONFIG)
    config.setdefault("volatility_regimes", VOLATILITY_REGIMES)
    config.setdefault(
        "max_contracts_per_symbol",
        config.get("max_contracts_per_trade", 5),
    )
    return config


@dataclass
class PositionSize:
    """Position size recommendation."""
    contracts: int
    dollar_amount: float
    risk_dollar_amount: float
    risk_percent: float
    kelly_fraction: float
    confidence_multiplier: float
    volatility_multiplier: float
    reason: str


class KellyCriterionSizer:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Kelly Formula:
    f* = (p * b - q) / b
    
    Where:
    - f* = fraction of capital to risk
    - p = probability of profit
    - q = probability of loss (1 - p)
    - b = ratio of profit to loss
    
    We use Fractional Kelly (0.25x) for safety.
    """
    
    def __init__(self):
        self.config = _get_position_sizer_config()
        self.logger = logging.getLogger(__name__)
        
    def calculate_kelly_fraction(
        self,
        probability_of_profit: float,
        profit_target_pct: float,
        stop_loss_pct: float,
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            probability_of_profit: P(profit) 0-1
            profit_target_pct: Expected profit % (e.g., 0.50 for 50%)
            stop_loss_pct: Stop loss % (e.g., 0.25 for 25%)
            
        Returns:
            Kelly fraction (0-1)
        """
        # Validate inputs
        if not (0 < probability_of_profit < 1):
            return 0.0
        
        p = probability_of_profit
        q = 1 - p
        b = profit_target_pct / stop_loss_pct  # Win/loss ratio
        
        # Kelly formula
        kelly = (p * b - q) / b
        
        # Apply fractional Kelly (0.25x default)
        fractional_kelly = kelly * self.config["kelly_fraction"]
        
        # Constrain to min/max
        kelly_min = self.config["kelly_min"]
        kelly_max = self.config["kelly_max"]
        
        return max(kelly_min, min(fractional_kelly, kelly_max))


class VolatilityAdjuster:
    """
    Adjust position size based on volatility regime.
    
    Logic:
    - EXTREME HIGH IV (>80): Reduce size to 0.5x
    - HIGH IV (60-80): Reduce size to 0.75x
    - NORMAL IV (40-60): Normal size 1.0x
    - LOW IV (20-40): Increase size to 1.25x
    - EXTREME LOW IV (<20): Increase size to 1.5x
    """
    
    def __init__(self):
        self.config = _get_position_sizer_config()
        
    def get_volatility_multiplier(self, iv_rank: Optional[float]) -> float:
        """
        Get size multiplier based on IV rank.
        
        Args:
            iv_rank: IV rank 0-100
            
        Returns:
            Multiplier (0.5x to 1.5x)
        """
        if iv_rank is None:
            return 1.0
        
        # Map to volatility regimes from config
        regimes = self.config["volatility_regimes"]
        
        for regime in regimes:
            if regime["min_iv_rank"] <= iv_rank <= regime["max_iv_rank"]:
                return regime["position_size_multiplier"]
        
        # Default to 1.0 if no match
        return 1.0


class MedallionPositionSizer:
    """
    Main position sizer combining all methods.
    
    Position Size Formula:
    contracts = (portfolio_value * risk_pct * confidence * vol_multiplier) / max_loss_per_contract
    
    Constraints:
    - Max 2% risk per trade
    - Max portfolio delta
    - Max contracts per symbol
    - Min contract = 1
    """
    
    def __init__(self):
        self.config = _get_position_sizer_config()
        self.logger = logging.getLogger(__name__)
        self.kelly_sizer = KellyCriterionSizer()
        self.vol_adjuster = VolatilityAdjuster()
        
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
        Calculate optimal position size.
        
        Args:
            portfolio_value: Total portfolio value ($)
            max_loss_per_contract: Max loss per contract ($)
            signal_confidence: Signal confidence 0-1
            probability_of_profit: P(profit) 0-1 (optional)
            iv_rank: IV rank 0-100 (optional)
            current_portfolio_delta: Current portfolio delta
            position_delta_per_contract: Delta per contract of new position
            
        Returns:
            PositionSize with recommendations
        """
        # Step 1: Calculate Kelly fraction
        if probability_of_profit is not None and probability_of_profit > 0:
            kelly_fraction = self.kelly_sizer.calculate_kelly_fraction(
                probability_of_profit=probability_of_profit,
                profit_target_pct=self.config["target_profit_pct"],
                stop_loss_pct=self.config["stop_loss_pct"],
            )
        else:
            # Default to fractional Kelly if no PoP provided
            kelly_fraction = self.config["kelly_fraction"]
        
        # Step 2: Get volatility multiplier
        volatility_multiplier = self.vol_adjuster.get_volatility_multiplier(iv_rank)
        
        # Step 3: Apply confidence scaling
        confidence_multiplier = signal_confidence
        
        # Step 4: Calculate base risk amount
        base_risk_pct = self.config["max_risk_per_trade_pct"]
        actual_risk_pct = base_risk_pct * kelly_fraction * confidence_multiplier * volatility_multiplier
        
        # Enforce maximum risk per trade
        actual_risk_pct = min(actual_risk_pct, base_risk_pct)
        
        risk_dollar_amount = portfolio_value * actual_risk_pct
        
        # Step 5: Calculate contracts
        if max_loss_per_contract <= 0:
            self.logger.warning("Invalid max_loss_per_contract, using minimum position")
            contracts = 1
        else:
            contracts = int(risk_dollar_amount / max_loss_per_contract)
        
        # Step 6: Apply constraints
        contracts = max(1, contracts)  # Minimum 1 contract
        contracts = min(contracts, self.config["max_contracts_per_symbol"])
        
        # Step 7: Check portfolio delta constraint
        new_portfolio_delta = current_portfolio_delta + (contracts * position_delta_per_contract)
        max_portfolio_delta = self.config["max_portfolio_delta"]
        
        if abs(new_portfolio_delta) > max_portfolio_delta:
            # Reduce contracts to stay within delta limit
            max_contracts_by_delta = int(
                (max_portfolio_delta - abs(current_portfolio_delta)) / abs(position_delta_per_contract)
            ) if position_delta_per_contract != 0 else contracts
            
            if max_contracts_by_delta < contracts:
                contracts = max(0, max_contracts_by_delta)
                self.logger.info(
                    f"Reduced contracts from {contracts} to {max_contracts_by_delta} due to delta constraint"
                )
        
        # Step 8: Calculate final amounts
        final_risk_dollar = contracts * max_loss_per_contract
        final_risk_pct = final_risk_dollar / portfolio_value if portfolio_value > 0 else 0
        dollar_amount = contracts * max_loss_per_contract  # Approximate
        
        # Build reason string
        reason = (
            f"Kelly: {kelly_fraction:.2%}, "
            f"Confidence: {confidence_multiplier:.2%}, "
            f"Vol Mult: {volatility_multiplier:.2f}x, "
            f"Risk: {final_risk_pct:.2%}"
        )
        
        return PositionSize(
            contracts=contracts,
            dollar_amount=dollar_amount,
            risk_dollar_amount=final_risk_dollar,
            risk_percent=final_risk_pct,
            kelly_fraction=kelly_fraction,
            confidence_multiplier=confidence_multiplier,
            volatility_multiplier=volatility_multiplier,
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
