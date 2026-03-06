"""
core/config.py
==============
Central configuration for the ATNN Quant Powerhouse trading system.

All configuration is expressed as Python dataclasses. Values are loaded from
environment variables with sensible defaults so the system works out-of-the-box
in paper mode without any configuration.

Usage
-----
    from core.config import get_config, SystemConfig

    cfg = get_config()
    print(cfg.risk.max_position_pct)  # 0.05
    print(cfg.alpaca.base_url)        # https://paper-api.alpaca.markets

Environment Variables
---------------------
    ALPACA_API_KEY        — Alpaca API key (also accepts APCA_API_KEY_ID)
    ALPACA_API_SECRET     — Alpaca secret key (also accepts APCA_API_SECRET_KEY)
    ALPACA_BASE_URL       — Alpaca base URL (default: paper trading endpoint)
    SYSTEM_MODE           — backtest | paper | live  (default: paper)
    LOG_LEVEL             — DEBUG | INFO | WARNING | ERROR  (default: INFO)
    TIMEZONE              — IANA timezone string  (default: America/New_York)
    PORTFOLIO_VALUE       — Initial portfolio value in USD (default: 100000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Top-50 S&P 500 equities by market cap (approximate, as of early 2026)
# together with the three ETF benchmarks used throughout the system.
# ---------------------------------------------------------------------------
_DEFAULT_SYMBOLS: List[str] = [
    # Benchmarks
    "SPY", "QQQ", "IWM",
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT",
    # Consumer / Industrials
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE",
    # Energy & Materials
    "XOM", "CVX", "COP",
    # Communication & Media
    "NFLX", "DIS", "CMCSA",
    # Semis / Hardware
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX",
    # Cloud / SaaS
    "CRM", "ORCL", "ADBE", "NOW",
    # Miscellaneous
    "RTX", "CAT", "DE", "LIN",
]

_DEFAULT_TIMEFRAMES: List[str] = ["1Day", "1Hour", "15Min"]


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

@dataclass
class AlpacaConfig:
    """Alpaca brokerage / market-data configuration.

    Keys are loaded from the environment. Both the legacy APCA_* naming
    convention and the newer ALPACA_* naming convention are supported; the
    ALPACA_* names take precedence.
    """

    api_key: str = field(default_factory=lambda: (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
        or ""
    ))
    secret_key: str = field(default_factory=lambda: (
        os.environ.get("ALPACA_API_SECRET")
        or os.environ.get("APCA_API_SECRET_KEY")
        or ""
    ))
    base_url: str = field(default_factory=lambda: (
        os.environ.get("ALPACA_BASE_URL")
        or os.environ.get("APCA_API_BASE_URL")
        or "https://paper-api.alpaca.markets"
    ))
    data_url: str = "https://data.alpaca.markets"
    paper: bool = field(default_factory=lambda: (
        os.environ.get("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
    ))

    def is_configured(self) -> bool:
        """Return True when both API credentials are non-empty."""
        return bool(self.api_key and self.secret_key)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Market-data pipeline configuration."""

    provider: str = field(default_factory=lambda: (
        os.environ.get("DATA_PROVIDER", "alpaca")
    ))
    symbols: List[str] = field(default_factory=lambda: list(_DEFAULT_SYMBOLS))
    timeframes: List[str] = field(default_factory=lambda: list(_DEFAULT_TIMEFRAMES))
    history_days: int = field(default_factory=lambda: int(
        os.environ.get("HISTORY_DAYS", "756")  # ~3 trading years
    ))
    # Minimum bars required before any strategy may emit signals
    min_history_bars: int = 60
    # yfinance fallback is allowed in backtest mode only
    allow_yfinance_fallback: bool = True
    # Cache directory for downloaded data
    cache_dir: str = field(default_factory=lambda: (
        os.environ.get("DATA_CACHE_DIR", "data/cache")
    ))


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class StatArbConfig:
    """Pairs / statistical-arbitrage strategy parameters."""

    entry_z: float = 1.5          # Enter when spread Z-score exceeds this
    exit_z: float = 0.3           # Exit when spread Z-score falls below this
    stop_z: float = 3.5           # Hard stop at this Z-score
    min_entry_z: float = 1.0      # Minimum Z-score to trigger entry
    lookback_days: int = 252      # 1 year minimum cointegration history
    # Kalman filter noise parameters (initial defaults)
    kalman_transition_cov: float = 1e-5
    kalman_observation_cov: float = 1e-3


@dataclass
class MomentumConfig:
    """Cross-sectional momentum strategy parameters."""

    lookback_days: int = 252      # 12-month lookback
    skip_days: int = 21           # Skip most recent month (1-month reversal)
    long_pct: float = 0.20        # Long top quintile
    short_pct: float = 0.20       # Short bottom quintile
    # Sector-neutral construction
    sector_neutral: bool = True
    # Volatility scaling: inverse-vol weight positions
    vol_scale: bool = True
    vol_target: float = 0.20      # Annualised volatility target


@dataclass
class FactorModelConfig:
    """Multi-factor alpha model parameters."""

    lookback_days: int = 63       # ~3 months for factor estimation
    # Factor composite Z-score thresholds
    entry_z: float = 0.75
    exit_z: float = -0.25
    # Factor weights (equal by default; override to time factors)
    quality_weight: float = 0.25
    value_weight: float = 0.25
    low_vol_weight: float = 0.25
    momentum_weight: float = 0.25


@dataclass
class StrategyConfig:
    """Aggregated strategy configuration."""

    stat_arb: StatArbConfig = field(default_factory=StatArbConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    factor_model: FactorModelConfig = field(default_factory=FactorModelConfig)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """Portfolio risk limits and position-sizing parameters."""

    # --- Position limits ---
    max_position_pct: float = field(default_factory=lambda: float(
        os.environ.get("RISK_MAX_POSITION_PCT", "0.15")
    ))
    max_sector_pct: float = field(default_factory=lambda: float(
        os.environ.get("RISK_MAX_SECTOR_PCT", "0.35")
    ))

    # --- Drawdown gates ---
    max_drawdown_halt: float = field(default_factory=lambda: float(
        os.environ.get("RISK_MAX_DRAWDOWN_HALT", "-0.20")
    ))
    max_drawdown_reduce: float = field(default_factory=lambda: float(
        os.environ.get("RISK_MAX_DRAWDOWN_REDUCE", "-0.15")
    ))
    # How far to reduce exposure when max_drawdown_reduce is breached
    drawdown_reduce_target: float = 0.50  # 50 % of normal exposure

    # --- Daily P&L gate ---
    daily_loss_limit: float = field(default_factory=lambda: float(
        os.environ.get("RISK_DAILY_LOSS_LIMIT", "-0.03")
    ))

    # --- Correlation limit ---
    max_correlation: float = field(default_factory=lambda: float(
        os.environ.get("RISK_MAX_CORRELATION", "0.85")
    ))
    # Lookback for pairwise correlation estimation
    correlation_lookback: int = 63  # ~3 months

    # --- Short-selling limits ---
    # Maximum gross short exposure as fraction of equity
    max_short_exposure: float = field(default_factory=lambda: float(
        os.environ.get("RISK_MAX_SHORT_EXPOSURE", "0.20")
    ))
    # Maximum individual short position as fraction of equity
    max_short_position_pct: float = field(default_factory=lambda: float(
        os.environ.get("RISK_MAX_SHORT_POSITION_PCT", "0.05")
    ))

    # --- Kelly criterion ---
    # Position size is capped at half-Kelly
    kelly_fraction: float = 0.5


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------

@dataclass
class LightGBMParams:
    """LightGBM hyperparameters (shared across prediction horizons)."""

    max_depth: int = 6
    num_leaves: int = 31
    learning_rate: float = 0.05
    min_child_samples: int = 50
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42


@dataclass
class MLConfig:
    """Machine-learning pipeline configuration."""

    feature_lookback: int = field(default_factory=lambda: int(
        os.environ.get("ML_FEATURE_LOOKBACK", "252")
    ))
    # Retraining frequency in calendar days
    retrain_freq_days: int = field(default_factory=lambda: int(
        os.environ.get("ML_RETRAIN_FREQ_DAYS", "7")
    ))
    # Training window in trading days (~2 years)
    train_window_days: int = 504
    # Prediction horizons in trading days
    horizons: List[int] = field(default_factory=lambda: [1, 5, 20])
    model_params: LightGBMParams = field(default_factory=LightGBMParams)
    # Directory for persisting trained models
    model_dir: str = field(default_factory=lambda: (
        os.environ.get("ML_MODEL_DIR", "models/lgbm")
    ))


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Walk-forward back-test configuration."""

    slippage_bps: float = field(default_factory=lambda: float(
        os.environ.get("BACKTEST_SLIPPAGE_BPS", "7")
    ))
    commission_per_share: float = field(default_factory=lambda: float(
        os.environ.get("BACKTEST_COMMISSION_PER_SHARE", "0.005")
    ))
    # Market-impact model: cost ≈ market_impact_factor * sqrt(qty / adv)
    market_impact_factor: float = 0.1
    # Short borrow rate, annualised
    short_borrow_rate: float = 0.02

    # Walk-forward parameters
    train_window: int = field(default_factory=lambda: int(
        os.environ.get("BACKTEST_TRAIN_WINDOW", "504")  # 2 trading years
    ))
    test_window: int = field(default_factory=lambda: int(
        os.environ.get("BACKTEST_TEST_WINDOW", "21")  # 1 trading month
    ))
    step_size: int = field(default_factory=lambda: int(
        os.environ.get("BACKTEST_STEP_SIZE", "21")
    ))
    min_windows: int = field(default_factory=lambda: int(
        os.environ.get("BACKTEST_MIN_WINDOWS", "12")
    ))

    # CPCV parameters
    cpcv_groups: int = 10
    cpcv_purge_days: int = 25  # max holding period (20) + 5-day buffer
    cpcv_min_sharpe_pct: float = 0.80  # 80 % of paths must show positive Sharpe


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """Top-level system / runtime configuration."""

    mode: str = field(default_factory=lambda: (
        os.environ.get("SYSTEM_MODE", "paper").lower()
    ))
    log_level: str = field(default_factory=lambda: (
        os.environ.get("LOG_LEVEL", "INFO").upper()
    ))
    timezone: str = field(default_factory=lambda: (
        os.environ.get("TIMEZONE", "America/New_York")
    ))
    initial_portfolio_value: float = field(default_factory=lambda: float(
        os.environ.get("PORTFOLIO_VALUE", "100000")
    ))
    log_dir: str = field(default_factory=lambda: (
        os.environ.get("LOG_DIR", "logs")
    ))
    # Session ID is set at runtime by the orchestrator; config holds default.
    session_id: Optional[str] = None

    def validate(self) -> None:
        """Raise ValueError if the mode is not a recognised value."""
        valid_modes = {"backtest", "paper", "live"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid SYSTEM_MODE={self.mode!r}. "
                f"Must be one of {sorted(valid_modes)}."
            )


# ---------------------------------------------------------------------------
# Root config container
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Fully-populated system configuration.

    Compose all sub-configs into a single object.  Callers should use
    :func:`get_config` rather than instantiating this class directly.
    """

    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def validate(self) -> None:
        """Run validation on all sub-configs that support it."""
        self.system.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the config to a plain dictionary (for logging / audit)."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_CONFIG_CACHE: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Return the singleton :class:`Config` instance.

    The config is loaded once from the environment and cached.  Pass
    ``reload=True`` to force a fresh read (useful in tests or when env vars
    change at runtime).

    Parameters
    ----------
    reload:
        Force re-construction of the config from the current environment.

    Returns
    -------
    Config
        Fully-populated configuration object.

    Raises
    ------
    ValueError
        If any sub-config fails validation.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is None or reload:
        cfg = Config()
        cfg.validate()
        _CONFIG_CACHE = cfg

    return _CONFIG_CACHE
