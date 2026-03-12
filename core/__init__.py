"""
core/__init__.py
================
Public API for the ATNN Quant Powerhouse core infrastructure.

Importing from this package gives access to the four foundational modules:

    from core import get_config, get_trade_logger, RegimeDetector, RiskManager

"""

from core.config import (
    IBKRConfig,
    BacktestConfig,
    Config,
    DataConfig,
    FactorModelConfig,
    LightGBMParams,
    MLConfig,
    MomentumConfig,
    RiskConfig,
    StatArbConfig,
    StrategyConfig,
    SystemConfig,
    get_config,
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
    # config
    "IBKRConfig",
    "BacktestConfig",
    "Config",
    "DataConfig",
    "FactorModelConfig",
    "LightGBMParams",
    "MLConfig",
    "MomentumConfig",
    "RiskConfig",
    "StatArbConfig",
    "StrategyConfig",
    "SystemConfig",
    "get_config",
    # logger
    "TradeLogger",
    "get_trade_logger",
    "EVENT_SIGNAL",
    "EVENT_ORDER",
    "EVENT_FILL",
    "EVENT_RISK_EVENT",
    "EVENT_REGIME_CHANGE",
    "EVENT_PERF_SNAPSHOT",
    # regime detector
    "RegimeDetector",
    "RegimeState",
    "Regime",
    "VIXLevel",
    # risk manager
    "RiskAction",
    "RiskManager",
    "PortfolioState",
    "TradeApproval",
]
