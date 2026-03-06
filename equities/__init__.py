"""
equities/__init__.py
====================
Equities trading engine for the ATNN Quant Powerhouse system.

Exports all public data models and provides convenient top-level imports.

Modules
-------
equities.models           — Shared dataclasses: Signal, Order, Position, etc.
equities.strategies       — Strategy implementations (stat_arb, momentum, factor_model)
equities.signal_generator — Unified signal orchestration pipeline
equities.execution        — Broker interface, SimulatedBroker, ExecutionManager
"""

from equities.models import (
    Signal,
    Order,
    Position,
    PortfolioState,
    Account,
    Pair,
    Fill,
)

__all__ = [
    "Signal",
    "Order",
    "Position",
    "PortfolioState",
    "Account",
    "Pair",
    "Fill",
]
