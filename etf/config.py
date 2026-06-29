"""
etf/config.py
=============
Central configuration for the ETF tactical-allocation engine.

Every parameter is economics-driven and documented. No magic constants.
All values can be overridden by environment variables for ops flexibility,
but the defaults are chosen to be *robust*, not curve-fit to one sample.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# ETF universe
# ---------------------------------------------------------------------------
# A diversified, highly liquid multi-asset universe. Diversification across
# *asset classes* (not just equity sectors) is what lets a momentum + risk
# parity engine rotate into something that is trending when equities are not
# (e.g. bonds or gold in 2008/2022). All tickers are >$1B AUM, tight spreads.
_DEFAULT_RISK_UNIVERSE: List[str] = [
    # US equity (broad)
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000 (small cap)
    # International equity
    "EFA",   # Developed ex-US
    "EEM",   # Emerging markets
    # US equity sectors (rotation breadth)
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health care
    "XLI",   # Industrials
    # Fixed income (the diversifier / crisis hedge)
    "TLT",   # 20+yr Treasuries
    "IEF",   # 7-10yr Treasuries
    "LQD",   # Investment-grade credit
    # Real assets
    "GLD",   # Gold
    "DBC",   # Broad commodities
]

# Risk-free / "cash" sleeve. When the trend/vol-target engine wants to be out
# of risk, capital parks here instead of sitting idle. Short-duration T-bills
# carry minimal duration risk and earn the front-end yield.
_DEFAULT_CASH_ASSET: str = "BIL"  # 1-3 month T-bill ETF

# Benchmark for relative-performance attribution (alpha/beta).
_DEFAULT_BENCHMARK: str = "SPY"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# IBKR connection
# ---------------------------------------------------------------------------
@dataclass
class IBKRConfig:
    """Interactive Brokers connection configuration (shared convention with
    the VRP engine so a single Gateway/TWS works for both)."""

    host: str = field(default_factory=lambda: os.environ.get("IBKR_HOST", "127.0.0.1"))
    # 4002 = IB Gateway paper, 4001 = IB Gateway live, 7497 = TWS paper, 7496 = TWS live
    port: int = field(default_factory=lambda: _env_int("IBKR_PORT", 4002))
    # Use a DIFFERENT client_id from the VRP engine so both can run concurrently.
    client_id: int = field(default_factory=lambda: _env_int("ETF_IBKR_CLIENT_ID", 7))
    account: str = field(default_factory=lambda: os.environ.get("IBKR_ACCOUNT", ""))
    timeout: int = 30
    readonly: bool = False
    # IBKR market-data type for live quotes:
    #   1 = real-time (requires a paid/level-1 market-data subscription),
    #   2 = frozen (last close while market closed),
    #   3 = delayed (free, ~15-min lag),
    #   4 = delayed-frozen.
    # A fresh paper account usually has NO real-time subscription, so quote
    # fields come back NaN under type 1 and the rebalance aborts fail-safe.
    # Default to 3 (delayed) so the engine works out-of-the-box; set
    # IBKR_MARKET_DATA_TYPE=1 once a real-time subscription is active.
    market_data_type: int = field(
        default_factory=lambda: _env_int("IBKR_MARKET_DATA_TYPE", 3))


# ---------------------------------------------------------------------------
# Signal parameters
# ---------------------------------------------------------------------------
@dataclass
class SignalConfig:
    """Momentum + trend signal parameters.

    Lookbacks are in *trading days* (~21/month). The blend of multiple
    horizons (3/6/12 month) is intentionally robust: any single lookback is
    fragile, but their average is stable out-of-sample (Hurst, Ooi & Pedersen
    2017, "A Century of Evidence on Trend-Following Investing").
    """

    # Cross-sectional momentum lookbacks (trading days) and their blend weights.
    # 3m/6m/12m is the canonical robust set; we de-emphasise the noisiest 3m.
    momentum_lookbacks: List[int] = field(default_factory=lambda: [63, 126, 252])
    momentum_weights: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])

    # Trend filter: long only when price > SMA(trend_sma) AND 12-1 momentum > 0.
    # 200-day SMA is the canonical, regime-robust trend gate (Faber 2007).
    trend_sma: int = 200
    # 12-1 month momentum (skip the most recent month to avoid short-term
    # reversal contamination — Jegadeesh & Titman).
    ts_momentum_long: int = 252
    ts_momentum_skip: int = 21

    # Number of top-ranked risky ETFs to hold. Holding a handful (not 1)
    # diversifies idiosyncratic ETF risk while keeping conviction.
    top_k: int = 5

    # Realised-vol lookback for inverse-vol weighting (trading days).
    vol_lookback: int = 63


# ---------------------------------------------------------------------------
# Mean-reversion sleeve parameters (Sleeve B)
# ---------------------------------------------------------------------------
@dataclass
class MeanReversionConfig:
    """Short-horizon mean-reversion ("buy the dip") sleeve parameters.

    Connors-style RSI(2) oversold entries, restricted to broad equity ETFs that
    are still in a long-term uptrend (price > long SMA). This earns when markets
    chop and pull back — structurally low/negative correlation to the trend
    sleeve, which is exactly why it diversifies the book.

    References: Connors & Alvarez, *Short Term Trading Strategies That Work*
    (RSI(2) mean reversion); the long-term-trend gate avoids catching falling
    knives during structural bear markets.
    """

    # ETFs the sleeve may trade. Mean reversion is reliable on broad,
    # liquid equity indices/sectors; it is NOT applied to bonds/commodities.
    universe: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLI", "EFA",
    ])
    # RSI lookback (Connors uses 2 — captures sharp, short pullbacks).
    rsi_period: int = 2
    # Enter/hold while RSI below this (deeply oversold).
    # Tuned via bounded sweep around baseline; 8.0 gave the best Sharpe/CAGR
    # uplift in current long-horizon evidence without adding model complexity.
    rsi_oversold: float = 8.0
    # Long-term trend gate: only buy dips when price > SMA(trend_sma).
    trend_sma: int = 200
    # Max ETFs held at once (concentration control).
    max_positions: int = 3
    # Realised-vol lookback for inverse-vol weighting within the sleeve.
    vol_lookback: int = 21
    # Per-name cap inside the sleeve.
    max_position_weight: float = 0.40
    # Fraction of the sleeve's capital deployed when signals fire (rest cash).
    # <1.0 keeps the sleeve from being fully invested on a single noisy day.
    deploy_fraction: float = 1.0


# ---------------------------------------------------------------------------
# Defensive-carry sleeve parameters (Sleeve C)
# ---------------------------------------------------------------------------
@dataclass
class DefensiveCarryConfig:
    """Absolute (time-series) momentum on *defensive* assets only.

    The universe is deliberately equity-free (duration + credit + gold), so the
    sleeve is structurally non-equity-beta. It holds a defensive asset only when
    it is both trending up (price > long SMA) AND has positive trailing absolute
    momentum — Antonacci-style dual confirmation. This earns during the
    equity-momentum dead zones / flights-to-safety where the trend and
    mean-reversion (equity) sleeves stall, which is exactly the diversification
    Phase 2 needs to lift blended Sharpe.

    References: Antonacci, *Dual Momentum Investing* (absolute momentum gate);
    Asness/Moskowitz/Pedersen on time-series momentum across asset classes.
    """

    # Defensive universe (NO equities — that is the whole point).
    universe: List[str] = field(default_factory=lambda: ["TLT", "IEF", "LQD", "GLD"])
    # Absolute-momentum lookback (trading days). 84 (~4 months) is currently
    # more responsive and improved blended book Sharpe in bounded OOS tests.
    momentum_lookback: int = 84
    # Skip the most recent month (short-term reversal hygiene).
    momentum_skip: int = 21
    # Long-term trend gate.
    trend_sma: int = 200
    # Realised-vol lookback for inverse-vol weighting within the sleeve.
    vol_lookback: int = 63
    # Max defensive assets held at once.
    max_positions: int = 4
    # Per-name cap inside the sleeve.
    max_position_weight: float = 0.60
    # Fraction of sleeve capital deployed when signals fire (rest cash).
    deploy_fraction: float = 1.0


# ---------------------------------------------------------------------------
# Cross-sectional relative-strength sleeve parameters (Sleeve D)
# ---------------------------------------------------------------------------
@dataclass
class CrossSectionalConfig:
    """Dollar-neutral long/short cross-sectional relative-strength sleeve.

    Rank a broad cross-section of equity ETFs by *risk-adjusted* momentum
    (skip-month return / trailing vol), go long the strongest and short the
    weakest in equal-and-opposite notional. Because longs and shorts net to
    ~zero dollar exposure, the sleeve strips out market beta and isolates the
    relative-strength spread — structurally low-correlation to the long-only
    trend, mean-reversion, and defensive sleeves.

    Shorting is on liquid, hard-to-borrow-free index/sector ETFs only; collateral
    earns the cash rate (modeled via the backtester's cash leg). ETF-only.

    References: Jegadeesh & Titman (cross-sectional momentum); Asness et al.
    (risk-adjusted / "betting against beta" style ranking).
    """

    # Cross-section to rank (broad equity sectors + regions; liquid & shortable).
    universe: List[str] = field(default_factory=lambda: [
        "XLK", "XLF", "XLE", "XLV", "XLI", "QQQ", "IWM", "EFA", "EEM", "SPY",
    ])
    # Momentum lookback (trading days) and skip-month for reversal hygiene.
    momentum_lookback: int = 126
    momentum_skip: int = 21
    # Trailing-vol lookback used to risk-adjust the momentum score.
    vol_lookback: int = 63
    # Number of names on each side of the book.
    top_k: int = 3
    bottom_k: int = 3
    # Target GROSS exposure (long notional + |short notional|). 1.0 => 0.5 long
    # + 0.5 short, i.e. dollar-neutral with 100% gross.
    gross_target: float = 1.0
    # Per-name cap (as a fraction of one side's notional).
    max_position_weight: float = 0.50


# ---------------------------------------------------------------------------
# Turn-of-month seasonality sleeve parameters (Sleeve E)
# ---------------------------------------------------------------------------
@dataclass
class SeasonalityConfig:
    """Turn-of-month (ToM) calendar-seasonality sleeve on broad equity ETFs.

    One of the most durable, internationally-replicated equity anomalies: index
    returns cluster around the turn of the calendar month — roughly the last
    trading day of a month plus the first few of the next — driven by recurring
    monthly cash flows (salary/401(k) contributions, pension & fund rebalancing,
    dividend reinvestment, window dressing). The sleeve is long broad equity
    ONLY inside the ToM window and in cash otherwise.

    Why it diversifies: the signal is the CALENDAR, not price. It is therefore
    structurally orthogonal to the price-driven trend, mean-reversion, and
    defensive-carry sleeves — exactly the uncorrelated edge the Sharpe gate
    needs. Being in cash ~75% of the time also makes it a low-vol, low-turnover
    return source that the ERC combiner can lever up where it is diversifying.

    Causality: the hold signal for the NEXT session is computed purely from the
    DECISION date's calendar position — first ``first_trading_days`` trading days
    of the month (counted causally from the slice) OR the last
    ``last_calendar_days`` calendar days of the month (from the date alone). No
    future index access. A long-term trend gate (price > ``trend_sma`` SMA)
    keeps the sleeve out of equities during sustained downtrends.

    References: Ariel (1987); Lakonishok & Smidt (1988); McConnell & Xu (2008,
    "Equity Returns at the Turn of the Month") — the effect persists out-of-sample
    and is concentrated, not a data-mined artifact.
    """

    # Broad, liquid equity ETFs that capture the index-level flow effect.
    universe: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    # Hold the NEXT session if the decision date is within the first N trading
    # days of the month (captures the +1..+N "early month" leg causally).
    first_trading_days: int = 3
    # ...OR within the last N calendar days of the month (captures the month-end
    # "-1" leg from the date alone, fully causal).
    last_calendar_days: int = 3
    # Long-term trend gate: only deploy when price > SMA(trend_sma). Antonacci-
    # style crash protection; keeps the sleeve flat in bear markets.
    trend_sma: int = 200
    # Realised-vol lookback for inverse-vol weighting within the sleeve.
    vol_lookback: int = 63
    # Per-name cap inside the sleeve.
    max_position_weight: float = 0.60
    # Fraction of sleeve capital deployed when in the window (rest cash).
    deploy_fraction: float = 1.0


# ---------------------------------------------------------------------------
# Volatility-managed overlay parameters (Moreira–Muir conditional vol timing)
# ---------------------------------------------------------------------------
@dataclass
class VolManagedConfig:
    """Conditional volatility-timing overlay for an equity-beta sleeve.

    Moreira & Muir (2017, "Volatility-Managed Portfolios", JF) show that scaling
    a risky position *inversely* to its recent realised volatility raises the
    risk-adjusted return: volatility is strongly persistent and forecastable at
    short horizons, while expected returns are not, so cutting exposure into
    high-vol regimes (and adding in calm ones) harvests a *timing* premium and
    truncates the left tail (Harvey et al. 2018, "The Impact of Vol Targeting").

    This overlay is DISTINCT from the sleeve's own *unconditional* vol target
    (which uses a slow ~63d estimate to set the average risk level). Here we use
    a SHORT ``realized_window`` (≈1 month) on the sleeve's actual held basket to
    time exposure around that average. The combiner's book-level vol target then
    absorbs any residual level effect, isolating the timing benefit.

    Causality: realised vol is measured on the trailing window ending at the
    decision bar (the same prices the inner sleeve already saw); no future data.
    """

    # Apply the overlay (opt-in). OFF reproduces the bare sleeve bit-for-bit.
    enabled: bool = False
    # Short realised-vol lookback (trading days) — the conditional timing signal.
    realized_window: int = 20
    # Annualised vol the basket is scaled toward. Near a typical equity-sleeve
    # basket vol so the *average* scale ≈ 1 (pure timing, minimal level change).
    target_vol_annual: float = 0.12
    # Cap on how far the overlay may lever the sleeve UP in calm regimes.
    max_scale: float = 1.5
    # Floor on the overlay scale (never fully flatten on a single vol spike).
    min_scale: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio combiner parameters (Phase 3 — allocate capital ACROSS sleeves)
# ---------------------------------------------------------------------------
@dataclass
class PortfolioConfig:
    """Cross-sleeve capital-allocation (risk-parity) combiner.

    The combiner treats each sleeve as a sub-portfolio and allocates capital
    across them by **equal risk contribution (ERC)** — each sleeve contributes
    the same share of total portfolio risk — then scales the whole book to a
    volatility target. ERC is correlation-aware (unlike naive inverse-vol), so
    it auto-tilts toward whatever is currently diversifying and avoids letting a
    single low-vol-but-correlated sleeve dominate.

    All combiner weights are computed from a TRAILING covariance window lagged
    one day, so the allocation is strictly out-of-sample at every point.

    References: Maillard, Roncalli & Teïletché, "The Properties of Equally
    Weighted Risk Contribution Portfolios" (ERC); Spinu (convex formulation).
    """

    # Allocation method across sleeves.
    # Defaulting to equal-weight across sleeves is currently more robust in this
    # repository's long-horizon evidence than ERC/inverse-vol.
    method: str = field(default_factory=lambda: os.environ.get(
        "ETF_PORTFOLIO_METHOD", "equal"
    ))  # one of {"erc", "inverse_vol", "equal"}
    # Trailing window (trading days) for the sleeve covariance estimate.
    cov_lookback: int = 126
    # Combiner rebalance cadence (trading days). Monthly keeps cross-sleeve
    # turnover cost low; sleeve-internal cadences are unchanged.
    rebalance_every: int = 21
    # Annualised volatility target for the COMBINED book.
    target_volatility: float = 0.10
    # Hard cap on combined gross exposure. 1.0 = no leverage (Phase 3). Phase 4
    # raises this to convert the high Sharpe into CAGR under strict risk control.
    max_leverage: float = 1.0
    # Floor on the vol-target scale (never fully de-risk to zero on a vol spike).
    min_scale: float = 0.0
    # Idle (unallocated) capital earns the risk-free rate (money-market proxy).
    cash_earns_rf: bool = True

    # --- Phase 4 (return lever): leverage serves the risk target, never chases
    # return beyond it. All three controls below are OPT-IN so the Phase 3 book
    # (max_leverage=1.0, dd_derisk=False) is reproduced bit-for-bit. ---
    # Combined-book drawdown circuit-breaker. When ON, gross exposure scales
    # down linearly during book drawdowns using the SAME thresholds the sleeves
    # use (cfg.risk.dd_start / dd_full / dd_min_exposure) — capital preservation
    # so a *levered* book survives crises to keep compounding.
    dd_derisk: bool = False
    # Annual margin-interest SPREAD over the risk-free rate, charged daily on the
    # levered portion (gross - 1.0). ~150 bps approximates IBKR Pro tiered margin
    # on a liquid-ETF book; it is a real drag that any honest levered backtest
    # must subtract, so leverage only pays when the gross edge clears its cost.
    margin_spread_annual: float = 0.015


# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    """Portfolio risk controls: vol targeting, leverage caps, drawdown overlay,
    and per-position concentration limits."""

    # Annualised portfolio volatility target. 10% is a moderate, institution-
    # grade risk budget for a multi-asset book; vol-targeting both raises
    # risk-adjusted return and tames left-tail clustering (Barroso & Santa-Clara).
    target_volatility: float = field(default_factory=lambda: _env_float("ETF_TARGET_VOL", 0.10))

    # Hard cap on gross exposure. Default 1.0 = long-only, NO leverage (safest
    # for a retail account). Can be raised (e.g. 1.5) once live-validated.
    max_gross_leverage: float = field(default_factory=lambda: _env_float("ETF_MAX_LEVERAGE", 1.0))

    # Maximum weight in any single ETF (concentration cap).
    max_position_weight: float = 0.30

    # Drawdown de-risking overlay. When the strategy's own equity drawdown
    # breaches `dd_start`, exposure scales down linearly to `dd_min_exposure`
    # by the time drawdown reaches `dd_full`. This is a circuit-breaker, not a
    # signal — it preserves capital so the engine survives to compound.
    dd_start: float = 0.08          # begin de-risking at -8%
    dd_full: float = 0.20           # fully de-risked floor at -20%
    dd_min_exposure: float = 0.25   # never below 25% (avoid permanent cash trap)

    # Realised-vol estimation window for portfolio vol targeting.
    portfolio_vol_lookback: int = 42

    # --- Hard kill-switch levels (Phase 5 live safety) -------------------
    # These are CATASTROPHIC halts, distinct from the smooth dd_derisk overlay:
    # crossing them stops *all* new trading until a human resets. The overlay
    # gently de-risks; the kill-switch slams the brakes.
    hard_halt_drawdown: float = 0.25   # halt all trading at -25% book drawdown
    max_daily_loss: float = 0.08       # halt for the day at -8% single-day P&L


# ---------------------------------------------------------------------------
# Execution / cost parameters
# ---------------------------------------------------------------------------
@dataclass
class ExecutionConfig:
    """Trading frequency and realistic cost assumptions."""

    # Rebalance cadence in trading days. Monthly (21) keeps turnover/costs low
    # while staying responsive — daily rebalancing of a momentum book is eaten
    # alive by costs and adds little signal.
    rebalance_every: int = field(default_factory=lambda: _env_int("ETF_REBALANCE_DAYS", 21))

    # IBKR fixed-tier commission ~ $0.005/share (min $1). For weight-based
    # backtests we model cost as bps of traded notional, which is the honest
    # equivalent for liquid ETFs.
    commission_bps: float = 0.5     # ~0.5 bps round-trip-equivalent on notional
    # Slippage for liquid ETFs (penny-wide spreads). 2 bps is conservative.
    slippage_bps: float = 2.0

    # Don't trade a position if the target change is below this (avoid churn).
    min_rebalance_delta: float = 0.02  # 2% of NAV

    # Post-fill reconciliation tolerance: max acceptable per-symbol drift between
    # the realised live book and the target weights before a cycle is flagged as
    # a MISMATCH (which hard-blocks the next cycle until reviewed).
    #
    # This is DELIBERATELY distinct from and wider than ``min_rebalance_delta``.
    # ``min_rebalance_delta`` is a *churn* threshold (don't bother trading tiny
    # deltas); reconciliation tolerance is about *fill realism*. After a real
    # rebalance the realised book legitimately differs from target by a few %
    # because of: whole-share rounding, fills executing at live prices while the
    # book is valued on (possibly delayed) marks, and the equity basis shifting
    # between sizing and post-fill valuation. Using the 2% churn threshold here
    # flags a NORMAL, fully-established book as a mismatch and then permanently
    # self-blocks every subsequent cycle — a false positive that looks like the
    # bot "stopped trading". 5% still catches genuine failures (rejected orders,
    # large partial fills, zero-fill from a competing session) which drift the
    # book by far more than realistic fill noise.
    reconciliation_tolerance: float = field(default_factory=lambda: _env_float(
        "ETF_RECONCILIATION_TOLERANCE", 0.05))

    # Order type for live/paper execution: "MKT" or "LMT".
    order_type: str = "MKT"
    # Marketable-limit offset (bps) when order_type == "LMT".
    limit_offset_bps: float = 5.0

    # Seconds to wait for submitted orders to fill before reading realised fills
    # and reconciling. Without this, reconciliation runs before the broker has
    # updated positions and flags a SPURIOUS mismatch that then blocks the next
    # cycle. Liquid ETF market orders fill in well under this; the wait ends
    # early as soon as all orders are done.
    fill_timeout_seconds: float = field(default_factory=lambda: _env_float(
        "ETF_FILL_TIMEOUT_SECONDS", 30.0))

    # JSONL telemetry sink for per-cycle slippage (modeled-vs-realised fills).
    slippage_log: str = field(default_factory=lambda: os.environ.get(
        "ETF_SLIPPAGE_LOG", ".etf_telemetry/slippage.jsonl"))

    # Persistent equity-state file (running peak + start-of-day equity) that
    # feeds live drawdown / daily-P&L into the kill-switch across restarts.
    state_path: str = field(default_factory=lambda: os.environ.get(
        "ETF_STATE_PATH", ".etf_telemetry/equity_state.json"))

    # Persistent reconciliation-state file. The pre-trade kill-switch reads the
    # PRIOR cycle's reconciliation outcome from here and blocks a new cycle if it
    # left an unresolved mismatch (never trade on top of an inconsistent book).
    recon_state_path: str = field(default_factory=lambda: os.environ.get(
        "ETF_RECON_STATE_PATH", ".etf_telemetry/reconciliation_state.json"))

    # Persistent scheduler-state file (last successful rebalance date) so the
    # live runner only rebalances once per cadence even across restarts.
    schedule_state_path: str = field(default_factory=lambda: os.environ.get(
        "ETF_SCHEDULE_STATE_PATH", ".etf_telemetry/schedule_state.json"))


# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Backtest window and accounting assumptions."""

    start: str = field(default_factory=lambda: os.environ.get("ETF_BACKTEST_START", "2007-01-01"))
    end: Optional[str] = field(default_factory=lambda: os.environ.get("ETF_BACKTEST_END") or None)
    initial_capital: float = field(default_factory=lambda: _env_float("ETF_INITIAL_CAPITAL", 100_000.0))
    # Annual risk-free rate for Sharpe (front-end T-bill proxy).
    risk_free_rate: float = field(default_factory=lambda: _env_float("ETF_RISK_FREE", 0.03))


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class ETFConfig:
    """Top-level configuration aggregating all sub-configs and the universe."""

    risk_universe: List[str] = field(default_factory=lambda: list(_DEFAULT_RISK_UNIVERSE))
    cash_asset: str = field(default_factory=lambda: os.environ.get("ETF_CASH_ASSET", _DEFAULT_CASH_ASSET))
    benchmark: str = field(default_factory=lambda: os.environ.get("ETF_BENCHMARK", _DEFAULT_BENCHMARK))

    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    defensive_carry: DefensiveCarryConfig = field(default_factory=DefensiveCarryConfig)
    cross_sectional: CrossSectionalConfig = field(default_factory=CrossSectionalConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    vol_managed: VolManagedConfig = field(default_factory=VolManagedConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    @property
    def all_symbols(self) -> List[str]:
        """Every ticker the engine needs price data for (de-duplicated)."""
        syms = list(self.risk_universe)
        for extra in (list(self.mean_reversion.universe) + list(self.defensive_carry.universe)
                      + list(self.cross_sectional.universe) + list(self.seasonality.universe)
                      + [self.cash_asset, self.benchmark]):
            if extra and extra not in syms:
                syms.append(extra)
        return syms

    def validate(self) -> None:
        """Fail fast on internally inconsistent configuration."""
        s = self.signal
        if len(s.momentum_lookbacks) != len(s.momentum_weights):
            raise ValueError("momentum_lookbacks and momentum_weights must align")
        if abs(sum(s.momentum_weights) - 1.0) > 1e-6:
            raise ValueError("momentum_weights must sum to 1.0")
        if s.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if s.top_k > len(self.risk_universe):
            raise ValueError("top_k cannot exceed the risk universe size")
        r = self.risk
        if not (0 < r.target_volatility < 1.0):
            raise ValueError("target_volatility must be in (0, 1)")
        if r.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be > 0")
        if not (0 < r.max_position_weight <= 1.0):
            raise ValueError("max_position_weight must be in (0, 1]")
        if not (0 <= r.dd_start < r.dd_full):
            raise ValueError("require 0 <= dd_start < dd_full")
        if not (0 <= r.dd_min_exposure <= 1.0):
            raise ValueError("dd_min_exposure must be in [0, 1]")
        if not (0 < r.hard_halt_drawdown <= 1.0):
            raise ValueError("hard_halt_drawdown must be in (0, 1]")
        if r.hard_halt_drawdown <= r.dd_full:
            raise ValueError("hard_halt_drawdown must exceed dd_full (kill-switch sits beyond the de-risk floor)")
        if not (0 < r.max_daily_loss <= 1.0):
            raise ValueError("max_daily_loss must be in (0, 1]")
        mr = self.mean_reversion
        if mr.rsi_period < 1:
            raise ValueError("rsi_period must be >= 1")
        if not (0 < mr.rsi_oversold < 100):
            raise ValueError("rsi_oversold must be in (0, 100)")
        if mr.max_positions < 1:
            raise ValueError("mean_reversion.max_positions must be >= 1")
        if not (0 < mr.max_position_weight <= 1.0):
            raise ValueError("mean_reversion.max_position_weight must be in (0, 1]")
        if not (0 < mr.deploy_fraction <= 1.0):
            raise ValueError("deploy_fraction must be in (0, 1]")
        dc = self.defensive_carry
        if dc.momentum_lookback < 1:
            raise ValueError("defensive_carry.momentum_lookback must be >= 1")
        if dc.momentum_skip < 0:
            raise ValueError("defensive_carry.momentum_skip must be >= 0")
        if dc.max_positions < 1:
            raise ValueError("defensive_carry.max_positions must be >= 1")
        if not (0 < dc.max_position_weight <= 1.0):
            raise ValueError("defensive_carry.max_position_weight must be in (0, 1]")
        if not (0 < dc.deploy_fraction <= 1.0):
            raise ValueError("defensive_carry.deploy_fraction must be in (0, 1]")
        cs = self.cross_sectional
        if cs.momentum_lookback < 1:
            raise ValueError("cross_sectional.momentum_lookback must be >= 1")
        if cs.momentum_skip < 0:
            raise ValueError("cross_sectional.momentum_skip must be >= 0")
        if cs.top_k < 1 or cs.bottom_k < 1:
            raise ValueError("cross_sectional.top_k and bottom_k must be >= 1")
        if cs.top_k + cs.bottom_k > len(cs.universe):
            raise ValueError("cross_sectional top_k + bottom_k cannot exceed its universe size")
        if cs.gross_target <= 0:
            raise ValueError("cross_sectional.gross_target must be > 0")
        if not (0 < cs.max_position_weight <= 1.0):
            raise ValueError("cross_sectional.max_position_weight must be in (0, 1]")
        se = self.seasonality
        if se.first_trading_days < 0 or se.last_calendar_days < 0:
            raise ValueError("seasonality first_trading_days and last_calendar_days must be >= 0")
        if se.first_trading_days == 0 and se.last_calendar_days == 0:
            raise ValueError("seasonality must hold at least one ToM leg (first/last days both 0)")
        if se.trend_sma < 1:
            raise ValueError("seasonality.trend_sma must be >= 1")
        if se.vol_lookback < 1:
            raise ValueError("seasonality.vol_lookback must be >= 1")
        if not (0 < se.max_position_weight <= 1.0):
            raise ValueError("seasonality.max_position_weight must be in (0, 1]")
        if not (0 < se.deploy_fraction <= 1.0):
            raise ValueError("seasonality.deploy_fraction must be in (0, 1]")
        vm = self.vol_managed
        if vm.realized_window < 2:
            raise ValueError("vol_managed.realized_window must be >= 2")
        if not (0 < vm.target_vol_annual < 1.0):
            raise ValueError("vol_managed.target_vol_annual must be in (0, 1)")
        if vm.max_scale <= 0:
            raise ValueError("vol_managed.max_scale must be > 0")
        if not (0 <= vm.min_scale <= vm.max_scale):
            raise ValueError("vol_managed.min_scale must be in [0, max_scale]")
        p = self.portfolio
        if p.method not in ("erc", "inverse_vol", "equal"):
            raise ValueError("portfolio.method must be one of {erc, inverse_vol, equal}")
        if p.cov_lookback < 2:
            raise ValueError("portfolio.cov_lookback must be >= 2")
        if p.rebalance_every < 1:
            raise ValueError("portfolio.rebalance_every must be >= 1")
        if not (0 < p.target_volatility < 1.0):
            raise ValueError("portfolio.target_volatility must be in (0, 1)")
        if p.max_leverage <= 0:
            raise ValueError("portfolio.max_leverage must be > 0")
        if not (0 <= p.min_scale <= 1.0):
            raise ValueError("portfolio.min_scale must be in [0, 1]")
        if p.margin_spread_annual < 0:
            raise ValueError("portfolio.margin_spread_annual must be >= 0")


def get_default_config() -> ETFConfig:
    """Return a validated default configuration."""
    cfg = ETFConfig()
    cfg.validate()
    return cfg
