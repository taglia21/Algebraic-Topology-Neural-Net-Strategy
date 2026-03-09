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
    # Close at this fraction of max credit.
    # 0.45 = close when spread is worth 45% of credit received.
    # Alpha experiment: 0.45 outperformed 0.50 when combined with earlier
    # tight-profit window (21 DTE). Captures gains faster during the
    # steepest part of the theta decay curve (30-21 DTE), reducing
    # exposure to gamma risk in the final 2 weeks.
    # Reference: LinkedIn theta analysis shows 70% of credit spread
    # profit is captured in the first 24 DTE, with the last 30%
    # carrying disproportionate gamma risk.
    profit_target_pct: float = 0.45
    # Tighter profit target when DTE < this threshold.
    # Moved from 14 to 21 DTE: theta decay accelerates sharply
    # after 21 DTE (Schwab "sweet spot" research). Taking profits
    # earlier avoids the gamma acceleration zone.
    tight_profit_dte: int = 21
    tight_profit_pct: float = 0.65
    # Close if spread reaches this multiple of credit received (stop loss)
    # Tightened from 3.0x to 2.0x: audit showed avg stop loss = $-1,311 vs
    # avg win = $289 (payoff ratio 0.23). At 2.0x, max stop loss per contract
    # is ~2x credit, cutting avg loss roughly in half and improving payoff
    # ratio toward 0.5+. Backtested: reduces max DD from -53% to <-25%.
    stop_loss_multiple: float = 2.0
    # Days before expiry to force-close (don't hold to expiry)
    close_before_expiry_days: int = 3
    # Roll if ITM and DTE <= this
    roll_dte_threshold: int = 7

    # --- Sizing ---
    # Risk per trade as fraction of account.
    # Reduced from 25% to 22%: with 3 concurrent positions, per-trade
    # risk needs to be slightly lower to keep total portfolio risk in check.
    # At 22%, max risk per trade = $2,200 on $10K, keeping quantity at 1
    # contract for $1,500 spreads (15-pt width). As account grows past
    # ~$14K, the system can size to 2 contracts per position.
    risk_per_trade: float = 0.22
    # Maximum concurrent open spreads.
    # Increased from 2 to 3: baseline capital utilization was only 52.5%
    # with avg ~1.0 concurrent positions despite max of 2. Adding a 3rd
    # slot captures more VRP during contango periods. Backtest: 437 trades
    # (vs 178 baseline), all 6 years profitable, Calmar 0.92.
    max_concurrent_positions: int = 3
    # Maximum total risk as fraction of account.
    # Increased from 50% to 55%: with 3 positions at 22% risk each,
    # max theoretical = 66%, but capped at 55% as a safety valve.
    # This allows 2-3 simultaneous positions without hitting the ceiling.
    max_total_risk_pct: float = 0.55  # Max 55% of account at risk across all positions
    # XSP (mini-SPX) support: 1/10th size contracts for small accounts.
    # XSP multiplier is $100 vs SPX $100 per point — same multiplier,
    # but XSP trades in ~1/10th the strike range ($540 vs $5400).
    # Set to True to auto-detect based on account size, or force with env var.
    use_xsp: bool = False
    # Account threshold below which we auto-switch to XSP
    xsp_equity_threshold: float = 10_000.0


@dataclass
class VIXRegimeConfig:
    """VIX-based regime thresholds for trade entry and sizing.

    Based on the observation that VRP is richest when IV is elevated
    but not in crisis. Thin premium when VIX < 12 isn't worth the
    gamma risk. VIX > 35 signals potential regime break.
    """

    # No new trades when VIX below this (premium too thin).
    # Granular regime P&L (5-year audit, per-trade avg):
    #   VIX 14-16: ~$10/trade (marginal)    VIX 20-21: $97/trade (excellent)
    #   VIX 16-18: ~$20/trade (positive)    VIX 21-25: $76-92/trade (solid)
    #   VIX 18-20: $16-40/trade (modest)    VIX 27-35: $77-213/trade (best)
    #
    # Alpha experiment (2020-2025 backtest, 18 configs tested):
    #   VIX floor 16 + 3 concurrent + dynamic exits = +183.5% total return
    #   vs baseline +138.9%. All 6 years profitable (baseline had 2 losing).
    #   Sharpe 0.64 vs 0.55, Calmar 0.92 vs 0.79, MaxDD -20.6% vs -19.8%.
    #
    # Academic evidence (Predicting Alpha, Carr & Wu 2009): VRP persists at
    # all VIX levels. IV/RV ratio stays ~1.3x regardless of regime. The edge
    # is thinner but positive — and 548 dormant days at VIX 20 floor was the
    # single largest alpha leak identified.
    min_vix: float = 16.0
    # No new trades when VIX above this (tail risk)
    max_vix: float = 35.0
    # Standard sizing when VIX in this range
    standard_low: float = 20.0
    standard_high: float = 25.0
    # Sizing multiplier when VIX > standard_high
    # VIX 25-27 is a danger zone (panic transition, -$128/trade in audit).
    # VIX 27+ recovers nicely. Net: use 0.75x as modest cap.
    elevated_sizing_mult: float = 0.75
    # Reduced sizing multiplier when VIX in [min_vix, standard_low]
    # 0.35x in VIX 16-20 band: conservative enough to limit damage from
    # thin-premium environments, but captures the VRP that exists there.
    # Backtest: 437 trades at 77.8% WR, all 6 years profitable.
    low_vol_sizing_mult: float = 0.35
    # SPX must be above 200-day SMA for new trades (trend filter)
    require_uptrend: bool = False  # disabled — VRP works in all regimes, VIX filter is sufficient


@dataclass
class RiskConfig:
    """Portfolio-level risk management."""

    # Maximum drawdown before halting all trading — NO new trades.
    # Tightened from -30% to -15%: the old -30% threshold allowed the
    # account to spiral to -53% max DD. At -15%, we stop digging and
    # preserve capital for recovery. This is standard at prop desks.
    max_drawdown_halt: float = -0.15
    # Maximum drawdown before reducing position sizes to half
    max_drawdown_reduce: float = -0.10  # -10%
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
