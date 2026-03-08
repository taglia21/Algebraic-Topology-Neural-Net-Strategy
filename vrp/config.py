"""
vrp/config.py
=============
Central configuration for the VRP Alpha Engine.

All parameters are economics-driven. Every number has a reason.
No magic constants — each value traces to VRP research or IBKR constraints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IBKRConfig:
    """Interactive Brokers connection configuration."""

    host: str = field(default_factory=lambda: os.environ.get("IBKR_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("IBKR_PORT", "4002")))
    client_id: int = field(default_factory=lambda: int(os.environ.get("IBKR_CLIENT_ID", "1")))
    account: str = field(default_factory=lambda: os.environ.get("IBKR_ACCOUNT", ""))
    # 4001 = IB Gateway live, 4002 = IB Gateway paper
    # 7496 = TWS live, 7497 = TWS paper
    timeout: int = 30
    readonly: bool = False


@dataclass
class SpreadConfig:
    """SPX put credit spread parameters.

    These are calibrated to the VRP literature and $10K account constraints.

    Key references:
    - CBOE PutWrite Index methodology
    - Quantpedia VRP strategy (selling ATM straddles + OTM put protection)
    - Tastytrade mechanical selling at 45 DTE / 50% profit target
    """

    # --- Strike selection ---
    # Short put delta target (-0.15 = ~85% probability of profit)
    # Grid search across 972 combos shows -0.15 is optimal: richer premium,
    # higher Sharpe (1.15 vs 0.56), better compounding
    short_delta_target: float = -0.15
    # Acceptable delta range for short leg
    short_delta_min: float = -0.19
    short_delta_max: float = -0.11
    # Spread width in SPX points (15 pts = $1,500 max risk per contract)
    # 15-pt width maximizes compound returns for $10K account:
    # more contracts, faster compounding, dynamic widening in high-VIX
    spread_width: int = 15
    # Minimum credit as fraction of spread width (reject thin premium)
    min_credit_pct: float = 0.08  # At least $120 credit on $1,500 spread

    # --- Timing ---
    # Target days to expiration at entry
    target_dte: int = 42
    # Acceptable DTE range
    min_dte: int = 21
    max_dte: int = 56
    # Only open on specific days (None = any day)
    entry_days: Optional[list] = None  # e.g. [0, 2, 4] for Mon/Wed/Fri

    # --- Exit rules ---
    # Close at this fraction of max credit (50% = close when spread worth 50% of credit)
    # Grid search: 0.50 slightly beats 0.40 on Sharpe; 0.40 has higher PF (2.08)
    profit_target_pct: float = 0.50
    # Tighter profit target when DTE < this threshold
    tight_profit_dte: int = 14
    tight_profit_pct: float = 0.75
    # Close if spread reaches this multiple of credit received (stop loss)
    stop_loss_multiple: float = 3.0  # wider stop to avoid whipsaws
    # Days before expiry to force-close (don't hold to expiry)
    close_before_expiry_days: int = 3
    # Roll if ITM and DTE <= this
    roll_dte_threshold: int = 7

    # --- Sizing ---
    # Risk per trade as fraction of account
    risk_per_trade: float = 0.50  # Allow up to 50% of account per spread
    # Maximum concurrent open spreads
    # Grid search: 2 positions optimal (highest Sharpe 1.15, best Calmar)
    max_concurrent_positions: int = 2
    # Maximum total risk as fraction of account
    max_total_risk_pct: float = 1.00  # Allow up to 100%


@dataclass
class VIXRegimeConfig:
    """VIX-based regime thresholds for trade entry and sizing.

    Based on the observation that VRP is richest when IV is elevated
    but not in crisis. Thin premium when VIX < 12 isn't worth the
    gamma risk. VIX > 35 signals potential regime break.
    """

    # No new trades when VIX below this (premium too thin)
    # Raised to 14: backtest shows VIX <15 trades have negative expectation
    min_vix: float = 14.0
    # No new trades when VIX above this (tail risk)
    max_vix: float = 35.0
    # Standard sizing when VIX in this range
    standard_low: float = 15.0
    standard_high: float = 20.0
    # Increased sizing multiplier when VIX > standard_high
    elevated_sizing_mult: float = 1.5
    # Reduced sizing multiplier when VIX in [min_vix, standard_low] — half size
    low_vol_sizing_mult: float = 0.50
    # SPX must be above 200-day SMA for new trades (trend filter)
    require_uptrend: bool = False  # disabled — VRP works in all regimes, VIX filter is sufficient


@dataclass
class RiskConfig:
    """Portfolio-level risk management."""

    # Maximum drawdown before halting all trading
    max_drawdown_halt: float = -0.30  # -30% (loose enough to survive vol spikes)
    # Maximum drawdown before reducing position sizes
    max_drawdown_reduce: float = -0.20  # -20%
    # Daily P&L limit (halt if daily loss exceeds this)
    daily_loss_limit: float = -0.05  # -5%
    # Minimum account equity to trade (don't trade if below this)
    min_equity: float = 3000.0  # Allow trading down to $3K (1 mini spread)
    # Maximum portfolio delta (net SPX-equivalent delta)
    max_portfolio_delta: float = 15.0  # ~15 SPX delta points
    # Maximum portfolio vega exposure
    max_portfolio_vega: float = -500.0  # short vega limit


@dataclass
class BacktestConfig:
    """Options backtesting parameters."""

    # Commission per contract (IBKR rate for SPX options)
    commission_per_contract: float = 0.65
    # Slippage per contract in dollars (bid-ask spread cost)
    slippage_per_contract: float = 1.50
    # Initial capital
    initial_capital: float = 10_000.0
    # Risk-free rate for Sharpe calculation
    risk_free_rate: float = 0.05  # 5% (current T-bill rate approx)


@dataclass
class Config:
    """Root configuration container."""

    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    vix: VIXRegimeConfig = field(default_factory=VIXRegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    mode: str = field(default_factory=lambda: os.environ.get("VRP_MODE", "paper"))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.mode not in ("backtest", "paper", "live"):
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.spread.spread_width < 10:
            raise ValueError("Spread width must be >= 10 points")
        if self.spread.profit_target_pct <= 0 or self.spread.profit_target_pct >= 1:
            raise ValueError("Profit target must be between 0 and 1")
        if self.spread.stop_loss_multiple <= 1:
            raise ValueError("Stop loss multiple must be > 1")
        max_risk = self.spread.spread_width * 100  # dollars per contract
        if self.backtest.initial_capital < max_risk:
            raise ValueError(
                f"Initial capital ${self.backtest.initial_capital:,.0f} insufficient "
                f"for spread width {self.spread.spread_width} pts "
                f"(${max_risk:,.0f} max risk)"
            )


def get_config() -> Config:
    """Return a validated configuration instance."""
    cfg = Config()
    cfg.validate()
    return cfg
