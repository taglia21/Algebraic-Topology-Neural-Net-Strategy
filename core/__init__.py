"""
core/__init__.py
================
Public API for the ATNN v2 core infrastructure.
"""

from core.config import (
    # New v2 config classes
    ATNNConfig,
    SystemCfg,
    BrokerCfg,
    UniverseCfg,
    TDACfg,
    NNCfg,
    EnsembleCfg,
    RiskCfg,
    OptionsCfg,
    EquitiesCfg,
    BacktestCfg,
    ScheduleCfg,
    SmallAccountCfg,
    get_config,
    # Legacy aliases (backward compat — aliased to active classes)
    IBKRConfig,
    Config,
    DataConfig,
    RiskConfig,
    SystemConfig,
    BacktestConfig,
)
from core.logger import (
    TradeLogger,
    get_trade_logger,
    EVENT_SIGNAL,
    EVENT_ORDER,
    EVENT_FILL,
    EVENT_RISK_EVENT,
    EVENT_REGIME_CHANGE,
    EVENT_PERF_SNAPSHOT,
)
from core.regime_detector import (
    RegimeDetector,
    RegimeState,
    Regime,
    VIXLevel,
)
from core.risk_manager import (
    RiskAction,
    RiskManager,
    PortfolioState,
    TradeApproval,
)

__all__ = [
    # v2 config
    "ATNNConfig", "SystemCfg", "BrokerCfg", "UniverseCfg", "TDACfg",
    "NNCfg", "EnsembleCfg", "RiskCfg", "OptionsCfg", "EquitiesCfg",
    "BacktestCfg", "ScheduleCfg", "SmallAccountCfg",
    "get_config",
    # legacy config aliases (aliased to active classes)
    "IBKRConfig", "Config", "DataConfig", "RiskConfig", "SystemConfig",
    "BacktestConfig",
    # logger
    "TradeLogger", "get_trade_logger",
    "EVENT_SIGNAL", "EVENT_ORDER", "EVENT_FILL",
    "EVENT_RISK_EVENT", "EVENT_REGIME_CHANGE", "EVENT_PERF_SNAPSHOT",
    # regime detector
    "RegimeDetector", "RegimeState", "Regime", "VIXLevel",
    # risk manager
    "RiskAction", "RiskManager", "PortfolioState", "TradeApproval",
]
