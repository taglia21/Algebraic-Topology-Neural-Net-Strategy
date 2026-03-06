"""
equities/models.py
==================
Shared dataclasses for the equities trading engine.

All signal, order, position, and portfolio state representations are defined
here.  Every other equities module imports from this module — keeping the data
model in one place prevents circular imports and makes schema changes easy.

Classes
-------
Signal        — Strategy output: what to trade, direction, and strength.
Order         — Broker order with lifecycle status.
Fill          — Execution confirmation with fill price and quantity.
Position      — Current holding in a single symbol.
PortfolioState — Snapshot of total portfolio equity, cash, and positions.
Account       — Broker account metadata (buying power, margin, etc.).
Pair          — Cointegrated pair used by the statistical arbitrage strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A trading signal emitted by a strategy.

    Attributes
    ----------
    symbol:
        Primary ticker symbol this signal refers to.  For spread signals
        (stat arb), this is the *long leg* symbol; use ``metadata`` to
        carry the short leg.
    direction:
        ``"long"``  — open or add to a long position.
        ``"short"`` — open or add to a short position.
        ``"close"`` — close / flatten an existing position.
    strength:
        Normalised signal conviction in [0.0, 1.0].  1.0 = maximum
        conviction; 0.0 = no conviction (should not be emitted in practice).
    strategy:
        Human-readable strategy name, e.g. ``"stat_arb"``, ``"momentum"``.
    metadata:
        Arbitrary strategy-specific diagnostics such as z-scores, factor
        scores, hedge ratios, etc.
    timestamp:
        UTC timestamp when the signal was generated.
    """

    symbol: str
    direction: str          # 'long', 'short', 'close'
    strength: float         # 0.0 to 1.0
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "close"):
            raise ValueError(
                f"Signal.direction must be 'long', 'short', or 'close'; "
                f"got {self.direction!r}"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Signal.strength must be in [0, 1]; got {self.strength!r}"
            )


# ---------------------------------------------------------------------------
# Order / Fill
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """A broker order with full lifecycle state.

    Lifecycle: ``new`` → ``submitted`` → ``partially_filled``
               → ``filled`` | ``cancelled`` | ``rejected``

    Attributes
    ----------
    order_id:
        Unique identifier assigned by the execution manager.
    symbol:
        Ticker symbol.
    side:
        ``"buy"`` or ``"sell"``.
    qty:
        Requested number of shares (always positive).
    order_type:
        ``"market"`` or ``"limit"``.
    limit_price:
        Required when ``order_type == "limit"``; ignored otherwise.
    status:
        Current lifecycle status.
    fill_price:
        Average fill price; ``None`` until at least partially filled.
    fill_qty:
        Number of shares actually filled so far.
    created_at:
        UTC timestamp of order creation.
    strategy:
        Name of the strategy that generated this order (for attribution).
    signal_strength:
        Signal strength that drove this order (for position sizing).
    """

    order_id: str
    symbol: str
    side: str               # 'buy', 'sell'
    qty: int
    order_type: str         # 'market', 'limit'
    limit_price: Optional[float] = None
    status: str = "new"
    fill_price: Optional[float] = None
    fill_qty: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    strategy: str = ""
    signal_strength: float = 1.0

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Order.side must be 'buy' or 'sell'; got {self.side!r}")
        if self.order_type not in ("market", "limit"):
            raise ValueError(
                f"Order.order_type must be 'market' or 'limit'; got {self.order_type!r}"
            )
        if self.qty <= 0:
            raise ValueError(f"Order.qty must be positive; got {self.qty!r}")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")

    @property
    def is_filled(self) -> bool:
        """True when the order has been fully filled."""
        return self.status == "filled"

    @property
    def is_active(self) -> bool:
        """True when the order can still receive fills."""
        return self.status in ("new", "submitted", "partially_filled")

    @property
    def remaining_qty(self) -> int:
        """Shares not yet filled."""
        return self.qty - self.fill_qty


@dataclass
class Fill:
    """Execution confirmation for a (partial) order fill.

    Attributes
    ----------
    order_id:
        Matches the parent :class:`Order`.
    symbol:
        Ticker symbol.
    side:
        ``"buy"`` or ``"sell"``.
    fill_price:
        Actual execution price per share.
    fill_qty:
        Number of shares filled in this execution.
    slippage_bps:
        Slippage relative to arrival price, in basis points.
    timestamp:
        UTC timestamp of the fill.
    """

    order_id: str
    symbol: str
    side: str
    fill_price: float
    fill_qty: int
    slippage_bps: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Current holding in a single symbol.

    Attributes
    ----------
    symbol:
        Ticker symbol.
    qty:
        Net shares held.  Positive = long, negative = short.
    avg_entry:
        Volume-weighted average entry price.
    current_price:
        Latest mark-to-market price.
    unrealized_pnl:
        Mark-to-market P&L on the open position.
    sector:
        GICS sector for sector-exposure tracking.
    strategy:
        Strategy that originated this position.
    """

    symbol: str
    qty: int
    avg_entry: float
    current_price: float
    unrealized_pnl: float
    sector: str = "Unknown"
    strategy: str = ""

    @property
    def market_value(self) -> float:
        """Current market value of the position (positive = long)."""
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return self.qty * self.avg_entry


# ---------------------------------------------------------------------------
# Portfolio / Account
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Complete snapshot of portfolio equity, cash, and positions.

    Attributes
    ----------
    equity:
        Total mark-to-market portfolio value (cash + market values).
    cash:
        Undeployed cash balance.
    positions:
        Mapping of symbol → :class:`Position`.
    unrealized_pnl:
        Aggregate unrealised P&L across all open positions.
    realized_pnl:
        Cumulative realised P&L since inception.
    peak_equity:
        Highest equity level observed since inception.
    drawdown:
        Current drawdown from peak as a fraction (negative, e.g. -0.05 = −5%).
    """

    equity: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute position market values."""
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Sum of signed position market values (long - short)."""
        return sum(p.market_value for p in self.positions.values())


@dataclass
class Account:
    """Broker account metadata.

    Attributes
    ----------
    account_id:
        Broker account identifier.
    buying_power:
        Available buying power in USD.
    equity:
        Current total account equity.
    cash:
        Cash balance.
    pattern_day_trader:
        True if the account is flagged as a pattern day trader.
    """

    account_id: str
    buying_power: float
    equity: float
    cash: float
    pattern_day_trader: bool = False


# ---------------------------------------------------------------------------
# Pair (used by StatArbStrategy)
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    """A cointegrated pair of stocks found by the stat-arb strategy.

    Attributes
    ----------
    symbol_x:
        First symbol in the pair (the "y" variable in the regression).
    symbol_y:
        Second symbol in the pair (the "x" variable / hedge instrument).
    hedge_ratio:
        Current hedge ratio β such that spread = price_x − β × price_y.
    half_life:
        Estimated Ornstein-Uhlenbeck mean-reversion half-life in trading days.
    coint_pvalue:
        Engle-Granger cointegration test p-value (lower = stronger evidence).
    ou_theta:
        OU mean-reversion speed (θ).
    ou_mu:
        OU long-run mean (μ).
    ou_sigma:
        OU diffusion coefficient (σ).
    lookback_days:
        Number of days of history used for estimation.
    last_updated:
        UTC timestamp of last parameter estimation.
    """

    symbol_x: str
    symbol_y: str
    hedge_ratio: float
    half_life: float
    coint_pvalue: float
    ou_theta: float
    ou_mu: float
    ou_sigma: float
    lookback_days: int = 504
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def pair_id(self) -> str:
        """Canonical identifier: alphabetically sorted symbol pair."""
        a, b = sorted([self.symbol_x, self.symbol_y])
        return f"{a}/{b}"
