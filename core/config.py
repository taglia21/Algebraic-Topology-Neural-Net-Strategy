"""
core/config.py
==============
Central configuration for the ATNN v2 trading system.

Loads configuration from YAML files with environment variable overrides.
Provides typed access via nested dataclasses.

Usage
-----
    from core.config import get_config

    cfg = get_config()                              # loads config/default.yaml
    cfg = get_config("config/custom.yaml")          # custom YAML
    print(cfg.broker.host)                          # typed access
    print(cfg.risk.max_position_pct)

Environment Variable Overrides
------------------------------
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_ACCOUNT
    SYSTEM_MODE, LOG_LEVEL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default config file path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return as a dict."""
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Dataclass config sections
# ---------------------------------------------------------------------------

@dataclass
class SystemCfg:
    name: str = "ATNN-v2"
    mode: str = "paper"
    log_level: str = "INFO"
    data_dir: str = "./data"
    model_dir: str = "./models"

    def validate(self) -> None:
        valid_modes = {"backtest", "paper", "live"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode={self.mode!r}. Must be one of {sorted(valid_modes)}."
            )


@dataclass
class BrokerCfg:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account: str = ""
    timeout: int = 30
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5

    def is_configured(self) -> bool:
        return bool(self.account)


@dataclass
class UniverseCfg:
    symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA"
    ])
    benchmark: str = "SPY"


@dataclass
class TDACfg:
    ph_window: int = 30
    corr_window: int = 60
    diffusion_time: float = 1.0
    spectral_threshold: float = 1.0
    regime_lookback: int = 252


@dataclass
class NNCfg:
    model_type: str = "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    sequence_length: int = 60
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    direction_threshold: float = 0.005


@dataclass
class EnsembleCfg:
    default_tda_weight: float = 0.5
    default_nn_weight: float = 0.5
    min_signal_strength: float = 0.3
    agreement_bonus: float = 0.2
    disagreement_penalty: float = 0.3


@dataclass
class SmallAccountCfg:
    enabled: bool = True
    max_risk_per_trade: float = 50.0
    max_concurrent_positions: int = 3
    max_option_premium: float = 50.0


@dataclass
class RiskCfg:
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.20
    max_long_exposure: float = 1.0
    max_short_exposure: float = 0.5
    max_gross_exposure: float = 1.3
    kelly_fraction: float = 0.5
    daily_loss_reduce_pct: float = 0.03
    daily_loss_flatten_pct: float = 0.05
    max_drawdown_halt_pct: float = 0.15
    small_account: SmallAccountCfg = field(default_factory=SmallAccountCfg)


@dataclass
class OptionsCfg:
    enabled: bool = False
    strategies: List[str] = field(default_factory=lambda: ["vertical_spread", "iron_condor"])
    default_dte_range: List[int] = field(default_factory=lambda: [14, 45])
    profit_target_pct: float = 0.50
    early_close_dte: int = 2
    max_delta: float = 0.30
    preferred_underlyings: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])


@dataclass
class EquitiesCfg:
    enabled: bool = False
    fractional_shares: bool = False


@dataclass
class BacktestCfg:
    initial_capital: float = 444.0
    train_window: int = 756
    test_window: int = 21
    purge_gap: int = 5
    embargo_gap: int = 5
    commission_per_share: float = 0.005
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.001


@dataclass
class ScheduleCfg:
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "America/New_York"
    signal_time: str = "09:45"
    reconciliation_time: str = "15:45"


# ---------------------------------------------------------------------------
# Root config container
# ---------------------------------------------------------------------------

@dataclass
class ATNNConfig:
    """Root configuration for ATNN v2."""

    system: SystemCfg = field(default_factory=SystemCfg)
    broker: BrokerCfg = field(default_factory=BrokerCfg)
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    tda: TDACfg = field(default_factory=TDACfg)
    nn: NNCfg = field(default_factory=NNCfg)
    ensemble: EnsembleCfg = field(default_factory=EnsembleCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    options: OptionsCfg = field(default_factory=OptionsCfg)
    equities: EquitiesCfg = field(default_factory=EquitiesCfg)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)

    def validate(self) -> None:
        self.system.validate()

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


def _build_dataclass(cls, data: Dict[str, Any]):
    """Construct a dataclass from a dict, handling nested dataclasses."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            val = data[f.name]
            # Resolve the type for nested dataclasses
            ftype = f.type
            if isinstance(ftype, str):
                # Handle forward references
                ftype = globals().get(ftype, ftype)
            if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
                kwargs[f.name] = _build_dataclass(ftype, val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Override specific config values from environment variables."""
    env_map = {
        "IBKR_HOST": ("broker", "host"),
        "IBKR_PORT": ("broker", "port", int),
        "IBKR_CLIENT_ID": ("broker", "client_id", int),
        "IBKR_ACCOUNT": ("broker", "account"),
        "SYSTEM_MODE": ("system", "mode"),
        "LOG_LEVEL": ("system", "log_level"),
    }

    for env_var, path in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            section = path[0]
            key = path[1]
            converter = path[2] if len(path) > 2 else str
            raw.setdefault(section, {})[key] = converter(val)

    return raw


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CONFIG_CACHE: Optional[ATNNConfig] = None
_CACHED_PATH: Optional[str] = None


def get_config(
    config_path: Optional[str] = None,
    reload: bool = False,
) -> ATNNConfig:
    """Load and return the ATNN v2 configuration.

    Parameters
    ----------
    config_path:
        Path to a YAML config file. Defaults to ``config/default.yaml``.
    reload:
        Force re-read from disk.

    Returns
    -------
    ATNNConfig
    """
    global _CONFIG_CACHE, _CACHED_PATH

    path_str = config_path or str(_DEFAULT_CONFIG_PATH)

    if _CONFIG_CACHE is not None and not reload and _CACHED_PATH == path_str:
        return _CONFIG_CACHE

    path = Path(path_str)
    if path.exists():
        raw = _load_yaml(path)
    else:
        raw = {}

    # Apply env var overrides
    raw = _apply_env_overrides(raw)

    # Build typed config
    cfg = _build_dataclass(ATNNConfig, raw)
    cfg.validate()

    _CONFIG_CACHE = cfg
    _CACHED_PATH = path_str
    return cfg


# ---------------------------------------------------------------------------
# Backward compatibility — legacy code imports these names
# ---------------------------------------------------------------------------

# Re-export the old names so existing imports don't break.
# The old Config class with sub-configs is replaced, but we alias the key ones.
IBKRConfig = BrokerCfg
DataConfig = UniverseCfg
RiskConfig = RiskCfg
SystemConfig = SystemCfg
BacktestConfig = BacktestCfg
Config = ATNNConfig

# Legacy default symbols list (used by old main.py imports)
_DEFAULT_SYMBOLS: List[str] = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT",
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE",
    "XOM", "CVX", "COP",
    "NFLX", "DIS", "CMCSA",
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX",
    "CRM", "ORCL", "ADBE", "NOW",
    "RTX", "CAT", "DE", "LIN",
]

# Legacy dataclasses needed by old core/__init__.py imports
@dataclass
class StatArbConfig:
    entry_z: float = 1.5
    exit_z: float = 0.3
    stop_z: float = 3.5
    min_entry_z: float = 1.0
    lookback_days: int = 252
    kalman_transition_cov: float = 1e-5
    kalman_observation_cov: float = 1e-3


@dataclass
class MomentumConfig:
    lookback_days: int = 252
    skip_days: int = 21
    long_pct: float = 0.20
    short_pct: float = 0.20
    sector_neutral: bool = True
    vol_scale: bool = True
    vol_target: float = 0.20


@dataclass
class FactorModelConfig:
    lookback_days: int = 63
    entry_z: float = 0.75
    exit_z: float = -0.25
    quality_weight: float = 0.25
    value_weight: float = 0.25
    low_vol_weight: float = 0.25
    momentum_weight: float = 0.25


@dataclass
class MeanReversionConfig:
    lookback: int = 60
    entry_z: float = 1.2
    exit_z: float = 0.5
    hard_stop_z: float = 3.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rv_window: int = 20


@dataclass
class StrategyConfig:
    stat_arb: StatArbConfig = field(default_factory=StatArbConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    factor_model: FactorModelConfig = field(default_factory=FactorModelConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)


@dataclass
class LightGBMParams:
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
    feature_lookback: int = 252
    retrain_freq_days: int = 7
    train_window_days: int = 504
    horizons: List[int] = field(default_factory=lambda: [1, 5, 20])
    model_params: LightGBMParams = field(default_factory=LightGBMParams)
    model_dir: str = "models/lgbm"
