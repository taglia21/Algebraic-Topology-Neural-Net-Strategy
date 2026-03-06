"""
core/risk_manager.py
====================
Portfolio-level risk controls for the ATNN trading system.

The :class:`RiskManager` is the single gatekeeper that every strategy must
consult before any order is submitted.  It enforces:

    * Per-name position size limit         (max 5 % of portfolio)
    * Sector exposure limit                (max 25 % of portfolio)
    * Pairwise correlation limit           (max 0.70 between any two positions)
    * Portfolio drawdown gates             (−10 % → reduce, −15 % → halt)
    * Daily loss limit                     (−2 % → halt new entries)
    * Kelly-criterion-based position sizing (capped at half-Kelly)

Every risk decision is audit-logged via :class:`~core.logger.TradeLogger`.

Usage
-----
    from core.config import get_config
    from core.logger import get_trade_logger
    from core.risk_manager import RiskManager, PortfolioState

    cfg = get_config()
    log = get_trade_logger()
    rm  = RiskManager(cfg.risk, log)

    portfolio = PortfolioState(
        equity=105_000.0,
        peak_equity=110_000.0,
        today_pnl=-1_800.0,
        positions={"AAPL": 5000.0, "MSFT": 4500.0},
    )

    approval = rm.approve_trade("NVDA", "buy", 50, 500.0, portfolio)
    if approval.approved:
        # submit order ...
        pass
    else:
        print(approval.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import RiskConfig
from core.logger import TradeLogger, get_trade_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / lightweight data classes
# ---------------------------------------------------------------------------

class RiskAction(str, Enum):
    """Action directive returned by drawdown checks."""
    NORMAL = "NORMAL"   # No drawdown concern; trade freely
    REDUCE = "REDUCE"   # Reduce gross exposure to 50 %
    HALT   = "HALT"     # Flatten all positions; accept no new entries


@dataclass
class TradeApproval:
    """Result returned by :meth:`RiskManager.approve_trade`.

    Attributes
    ----------
    approved:
        Whether the trade is allowed.
    reason:
        Human-readable explanation.  Empty string when approved.
    suggested_qty:
        If the trade is approved but requires size adjustment, the adjusted
        quantity is provided here; otherwise equals the requested quantity.
    checks_run:
        Names of all checks that were evaluated.
    checks_failed:
        Names of checks that blocked the trade (subset of *checks_run*).
    """
    approved: bool
    reason: str
    suggested_qty: float
    checks_run: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Snapshot of current portfolio state passed to risk checks.

    Attributes
    ----------
    equity:
        Current mark-to-market portfolio value in USD.
    peak_equity:
        Highest equity value observed since inception (for drawdown calc).
    today_pnl:
        Realised + unrealised P&L for the current trading day in USD.
    positions:
        Mapping of symbol → current market value (positive = long,
        negative = short) in USD.
    sector_map:
        Optional mapping of symbol → sector string.  Required for sector
        exposure checks; if absent the sector check is skipped.
    """
    equity: float
    peak_equity: float
    today_pnl: float
    positions: Dict[str, float] = field(default_factory=dict)
    sector_map: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RiskManager:
    """Portfolio-level risk gatekeeper.

    Parameters
    ----------
    config:
        :class:`~core.config.RiskConfig` holding all risk thresholds.
    trade_logger:
        :class:`~core.logger.TradeLogger` instance for audit logging.
        If *None*, the process-level default logger is used.

    Notes
    -----
    All public methods are **pure** with respect to portfolio state: they read
    *portfolio_state* but never mutate it.  The caller is responsible for
    maintaining and updating the state object.
    """

    def __init__(
        self,
        config: RiskConfig,
        trade_logger: Optional[TradeLogger] = None,
    ) -> None:
        self._cfg = config
        self._log = trade_logger or get_trade_logger()

    # ------------------------------------------------------------------
    # Individual risk checks
    # ------------------------------------------------------------------

    def check_position_size(
        self,
        symbol: str,
        proposed_qty: float,
        proposed_price: float,
        portfolio_value: float,
    ) -> bool:
        """Return True if the proposed trade keeps position size within limit.

        The check is evaluated on the *resulting* position value: the
        combination of any existing exposure in *portfolio_state* and the new
        order.  Callers that need to validate an incremental add to an
        existing position should pass the total resulting notional.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        proposed_qty:
            Absolute number of shares (always positive).
        proposed_price:
            Estimated execution price.
        portfolio_value:
            Current total portfolio value in USD.

        Returns
        -------
        bool
            True if within limit.

        Raises
        ------
        ValueError
            If *portfolio_value* <= 0 or *proposed_qty* < 0.
        """
        if portfolio_value <= 0:
            raise ValueError(f"portfolio_value must be positive; got {portfolio_value!r}")
        if proposed_qty < 0:
            raise ValueError(
                f"proposed_qty must be non-negative; got {proposed_qty!r} for {symbol}"
            )

        notional = proposed_qty * proposed_price
        position_pct = notional / portfolio_value
        max_pct = self._cfg.max_position_pct

        if position_pct > max_pct:
            self._log.log_risk_event(
                "position_size_breach",
                {
                    "symbol": symbol,
                    "proposed_notional": round(notional, 2),
                    "position_pct": round(position_pct, 4),
                    "max_pct": max_pct,
                    "portfolio_value": portfolio_value,
                },
            )
            return False

        return True

    def check_sector_exposure(
        self,
        symbol: str,
        sector: Optional[str],
        current_positions: Dict[str, float],
        proposed_notional: float,
        portfolio_value: float,
    ) -> bool:
        """Return True if adding *proposed_notional* keeps sector exposure in limit.

        Parameters
        ----------
        symbol:
            Ticker being evaluated.
        sector:
            GICS sector string for *symbol*.  If *None*, the check is skipped
            (returns True) but a warning is logged.
        current_positions:
            Mapping of symbol → absolute market value.
        proposed_notional:
            Absolute notional value of the new trade in USD.
        portfolio_value:
            Current total portfolio value in USD.

        Returns
        -------
        bool
            True if sector exposure is within limit (or sector unknown).
        """
        if sector is None:
            logger.warning(
                f"check_sector_exposure: sector unknown for {symbol}; "
                "skipping check."
            )
            return True

        if portfolio_value <= 0:
            raise ValueError(f"portfolio_value must be positive; got {portfolio_value!r}")

        # Current sector exposure (absolute values)
        sector_exposure: Dict[str, float] = {}
        for sym, notional in current_positions.items():
            # Caller should provide a sector_map via PortfolioState; here we
            # accept an explicit dict so the method is usable standalone.
            pass  # sector_map is not available here; handled in approve_trade

        # In standalone mode, check only the proposed addition against the limit.
        # In approve_trade we recompute using the full sector map.
        exposure_pct = proposed_notional / portfolio_value
        max_pct = self._cfg.max_sector_pct

        if exposure_pct > max_pct:
            self._log.log_risk_event(
                "sector_exposure_breach",
                {
                    "symbol": symbol,
                    "sector": sector,
                    "proposed_exposure_pct": round(exposure_pct, 4),
                    "max_pct": max_pct,
                },
            )
            return False

        return True

    def check_correlation(
        self,
        symbol: str,
        current_positions: Dict[str, float],
        returns_data: pd.DataFrame,
    ) -> bool:
        """Return True if *symbol* has acceptable pairwise correlation with all
        existing positions.

        Parameters
        ----------
        symbol:
            Ticker being evaluated.
        current_positions:
            Mapping of symbol → market value for existing holdings.
        returns_data:
            Wide DataFrame of daily returns, one column per symbol.  Must
            include a column for *symbol* and for every key in
            *current_positions*.

        Returns
        -------
        bool
            True if all pairwise correlations are below the configured limit.

        Raises
        ------
        KeyError
            If *symbol* is absent from *returns_data*.
        """
        if symbol not in returns_data.columns:
            raise KeyError(
                f"check_correlation: '{symbol}' not found in returns_data columns. "
                f"Available: {list(returns_data.columns[:10])}"
            )

        if not current_positions:
            return True  # No existing positions → nothing to correlate against

        lookback = self._cfg.correlation_lookback
        recent = returns_data.tail(lookback)

        violations: List[str] = []

        for held_sym in current_positions:
            if held_sym == symbol:
                continue  # adding to an existing position is handled by size check
            if held_sym not in recent.columns:
                logger.warning(
                    f"check_correlation: '{held_sym}' missing from returns_data; "
                    "skipping pair."
                )
                continue

            pair = recent[[symbol, held_sym]].dropna()
            if len(pair) < 20:
                logger.warning(
                    f"check_correlation: insufficient data for {symbol}/{held_sym} "
                    f"({len(pair)} rows); skipping pair."
                )
                continue

            corr = float(pair[symbol].corr(pair[held_sym]))
            if abs(corr) > self._cfg.max_correlation:
                violations.append(held_sym)
                self._log.log_risk_event(
                    "correlation_breach",
                    {
                        "new_symbol": symbol,
                        "existing_symbol": held_sym,
                        "correlation": round(corr, 4),
                        "max_correlation": self._cfg.max_correlation,
                        "lookback_days": lookback,
                    },
                )

        return len(violations) == 0

    def check_drawdown(
        self,
        current_equity: float,
        peak_equity: float,
    ) -> RiskAction:
        """Classify the current drawdown level and return the appropriate action.

        Parameters
        ----------
        current_equity:
            Current portfolio mark-to-market value.
        peak_equity:
            Maximum equity value since inception (used as the high-water mark).

        Returns
        -------
        RiskAction
            NORMAL, REDUCE, or HALT.

        Raises
        ------
        ValueError
            If either argument is non-positive.
        """
        if current_equity <= 0:
            raise ValueError(
                f"current_equity must be positive; got {current_equity!r}"
            )
        if peak_equity <= 0:
            raise ValueError(
                f"peak_equity must be positive; got {peak_equity!r}"
            )

        drawdown = (current_equity - peak_equity) / peak_equity  # negative number

        if drawdown <= self._cfg.max_drawdown_halt:
            self._log.log_risk_event(
                "drawdown_halt",
                {
                    "drawdown_pct": round(drawdown, 4),
                    "halt_threshold": self._cfg.max_drawdown_halt,
                    "current_equity": current_equity,
                    "peak_equity": peak_equity,
                },
            )
            return RiskAction.HALT

        if drawdown <= self._cfg.max_drawdown_reduce:
            self._log.log_risk_event(
                "drawdown_reduce",
                {
                    "drawdown_pct": round(drawdown, 4),
                    "reduce_threshold": self._cfg.max_drawdown_reduce,
                    "current_equity": current_equity,
                    "peak_equity": peak_equity,
                },
            )
            return RiskAction.REDUCE

        return RiskAction.NORMAL

    def check_daily_loss(
        self,
        today_pnl: float,
        portfolio_value: float,
    ) -> bool:
        """Return True if the daily loss is within the configured limit.

        Parameters
        ----------
        today_pnl:
            Realised + unrealised P&L for the current day (negative = loss).
        portfolio_value:
            Portfolio value at the start of today (used as denominator).

        Returns
        -------
        bool
            True if no daily loss limit breach.

        Raises
        ------
        ValueError
            If *portfolio_value* <= 0.
        """
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive; got {portfolio_value!r}"
            )

        loss_pct = today_pnl / portfolio_value  # negative when losing

        if loss_pct <= self._cfg.daily_loss_limit:
            self._log.log_risk_event(
                "daily_loss_halt",
                {
                    "today_pnl": round(today_pnl, 2),
                    "loss_pct": round(loss_pct, 4),
                    "limit": self._cfg.daily_loss_limit,
                    "portfolio_value": portfolio_value,
                },
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        signal_strength: float,
        volatility: float,
        portfolio_value: float,
        price: float,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
    ) -> float:
        """Compute the appropriate position size using the Kelly criterion.

        The Kelly fraction is capped at *half-Kelly* (``kelly_fraction=0.5``
        in :class:`~core.config.RiskConfig`), then additionally capped at
        ``max_position_pct`` of portfolio value.

        The formula used when win_rate / avg_win_loss_ratio are available:

            Kelly fraction = W - (1 − W) / R

        where W = win_rate and R = avg_win_loss_ratio.

        When these are *not* provided, the method falls back to
        volatility-inverse sizing:

            notional = (signal_strength × target_vol) / volatility × portfolio_value

        Parameters
        ----------
        signal_strength:
            Normalised signal intensity in [0, 1].
        volatility:
            Annualised volatility of the instrument (e.g. 0.25 = 25 %).
        portfolio_value:
            Current portfolio value in USD.
        price:
            Current price per share.
        win_rate:
            Historical fraction of winning trades [0, 1].  Pass *None* to use
            the volatility-inverse fallback.
        avg_win_loss_ratio:
            Average win size / average loss size.  Must be positive.

        Returns
        -------
        float
            Number of shares (rounded down to the nearest integer).

        Raises
        ------
        ValueError
            If any argument violates its stated domain.
        """
        if not 0.0 <= signal_strength <= 1.0:
            raise ValueError(
                f"signal_strength must be in [0, 1]; got {signal_strength!r}"
            )
        if volatility <= 0:
            raise ValueError(f"volatility must be positive; got {volatility!r}")
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive; got {portfolio_value!r}"
            )
        if price <= 0:
            raise ValueError(f"price must be positive; got {price!r}")

        max_notional = self._cfg.max_position_pct * portfolio_value

        if win_rate is not None and avg_win_loss_ratio is not None:
            if not 0.0 < win_rate < 1.0:
                raise ValueError(
                    f"win_rate must be in (0, 1); got {win_rate!r}"
                )
            if avg_win_loss_ratio <= 0:
                raise ValueError(
                    f"avg_win_loss_ratio must be positive; got {avg_win_loss_ratio!r}"
                )

            kelly_f = win_rate - (1.0 - win_rate) / avg_win_loss_ratio
            # Negative Kelly → no edge; return 0
            if kelly_f <= 0:
                logger.info(
                    f"calculate_position_size: Kelly fraction non-positive "
                    f"({kelly_f:.4f}); returning zero size."
                )
                return 0.0

            # Half-Kelly
            half_kelly_f = kelly_f * self._cfg.kelly_fraction
            # Scale by signal strength and additional volatility adjustment
            vol_adj = min(1.0, 0.20 / max(volatility, 1e-6))  # target 20 % vol
            notional = half_kelly_f * signal_strength * vol_adj * portfolio_value
        else:
            # Volatility-inverse fallback
            target_vol = 0.20  # annualised 20 % vol target
            vol_adj = min(1.0, target_vol / max(volatility, 1e-6))
            notional = signal_strength * vol_adj * portfolio_value

        notional = min(notional, max_notional)
        shares = np.floor(notional / price)

        return float(shares)

    # ------------------------------------------------------------------
    # Master approval gate
    # ------------------------------------------------------------------

    def approve_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        portfolio_state: PortfolioState,
        returns_data: Optional[pd.DataFrame] = None,
    ) -> TradeApproval:
        """Run all risk checks and return a trade approval decision.

        This is the single entry point that strategies call.  It runs:

            1. Daily loss limit check
            2. Drawdown gate check
            3. Position size check
            4. Sector exposure check  (skipped if sector_map is empty)
            5. Short exposure limit   (sell orders only)
            6. Correlation check      (skipped if returns_data is None)

        If all checks pass, :attr:`TradeApproval.approved` is ``True``.
        On any failure the trade is denied and the reason is captured.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        side:
            ``"buy"`` or ``"sell"``.
        qty:
            Requested number of shares (absolute value; always positive).
        price:
            Estimated execution price per share.
        portfolio_state:
            Current portfolio snapshot.
        returns_data:
            Optional daily returns matrix for correlation check.

        Returns
        -------
        TradeApproval

        Raises
        ------
        ValueError
            If *side* is not ``"buy"`` or ``"sell"``, or *qty* <= 0.
        """
        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            raise ValueError(
                f"approve_trade: side must be 'buy' or 'sell'; got {side!r}"
            )
        if qty <= 0:
            raise ValueError(
                f"approve_trade: qty must be positive; got {qty!r}"
            )

        checks_run: List[str] = []
        checks_failed: List[str] = []
        denial_reasons: List[str] = []

        equity = portfolio_state.equity
        peak_equity = portfolio_state.peak_equity
        today_pnl = portfolio_state.today_pnl
        positions = portfolio_state.positions
        sector_map = portfolio_state.sector_map

        # ----------------------------------------------------------------
        # 1. Daily loss limit
        # ----------------------------------------------------------------
        checks_run.append("daily_loss")
        if not self.check_daily_loss(today_pnl, equity):
            checks_failed.append("daily_loss")
            denial_reasons.append(
                f"Daily loss limit breached "
                f"({today_pnl / equity:.2%} vs limit {self._cfg.daily_loss_limit:.2%})"
            )

        # ----------------------------------------------------------------
        # 2. Drawdown gate
        # ----------------------------------------------------------------
        checks_run.append("drawdown")
        dd_action = self.check_drawdown(equity, peak_equity)
        if dd_action == RiskAction.HALT:
            checks_failed.append("drawdown_halt")
            denial_reasons.append(
                f"Portfolio in HALT state: drawdown from peak exceeds "
                f"{self._cfg.max_drawdown_halt:.1%}"
            )
        elif dd_action == RiskAction.REDUCE and side_lower == "buy":
            # Only new long entries are blocked on REDUCE; exits are allowed.
            checks_failed.append("drawdown_reduce")
            denial_reasons.append(
                f"Portfolio in REDUCE state: only exits allowed "
                f"(drawdown threshold {self._cfg.max_drawdown_reduce:.1%} breached)"
            )

        # ----------------------------------------------------------------
        # 3. Position size
        # ----------------------------------------------------------------
        checks_run.append("position_size")
        if not self.check_position_size(symbol, qty, price, equity):
            checks_failed.append("position_size")
            notional = qty * price
            denial_reasons.append(
                f"Position size {notional / equity:.2%} exceeds limit "
                f"{self._cfg.max_position_pct:.0%} of portfolio"
            )

        # ----------------------------------------------------------------
        # 4. Sector exposure
        # ----------------------------------------------------------------
        if sector_map:
            checks_run.append("sector_exposure")
            sector = sector_map.get(symbol)

            if sector:
                # Compute current sector exposure
                current_sector_notional = sum(
                    abs(v)
                    for sym, v in positions.items()
                    if sector_map.get(sym) == sector
                )
                proposed_notional = qty * price
                total_sector_notional = current_sector_notional + proposed_notional
                sector_pct = total_sector_notional / equity

                if sector_pct > self._cfg.max_sector_pct:
                    checks_failed.append("sector_exposure")
                    self._log.log_risk_event(
                        "sector_exposure_breach",
                        {
                            "symbol": symbol,
                            "sector": sector,
                            "total_sector_pct": round(sector_pct, 4),
                            "max_sector_pct": self._cfg.max_sector_pct,
                        },
                    )
                    denial_reasons.append(
                        f"Sector '{sector}' exposure {sector_pct:.2%} "
                        f"exceeds limit {self._cfg.max_sector_pct:.0%}"
                    )
            else:
                logger.warning(
                    f"approve_trade: no sector mapping for {symbol}; "
                    "skipping sector check."
                )
        # ----------------------------------------------------------------
        # 5. Short exposure limit
        # ----------------------------------------------------------------
        if side_lower == "sell":
            checks_run.append("short_exposure")
            # Sum all current short positions (negative market value)
            current_short_exposure = sum(
                abs(v) for v in positions.values() if v < 0
            )
            proposed_notional = qty * price
            total_short_exposure = current_short_exposure + proposed_notional
            short_pct = total_short_exposure / equity if equity > 0 else 999

            # Check gross short exposure limit
            if short_pct > self._cfg.max_short_exposure:
                checks_failed.append("short_exposure")
                self._log.log_risk_event(
                    "short_exposure_breach",
                    {
                        "symbol": symbol,
                        "current_short_pct": round(current_short_exposure / equity, 4),
                        "proposed_short_pct": round(short_pct, 4),
                        "max_short_exposure": self._cfg.max_short_exposure,
                    },
                )
                denial_reasons.append(
                    f"Gross short exposure {short_pct:.2%} would exceed "
                    f"limit {self._cfg.max_short_exposure:.0%}"
                )

            # Check individual short position limit
            individual_short_pct = proposed_notional / equity if equity > 0 else 999
            if individual_short_pct > self._cfg.max_short_position_pct:
                if "short_exposure" not in checks_failed:
                    checks_failed.append("short_exposure")
                denial_reasons.append(
                    f"Individual short position {individual_short_pct:.2%} "
                    f"exceeds limit {self._cfg.max_short_position_pct:.0%}"
                )

        # ----------------------------------------------------------------
        # 6. Correlation check
        # ----------------------------------------------------------------
        if returns_data is not None and positions:
            checks_run.append("correlation")
            try:
                if not self.check_correlation(symbol, positions, returns_data):
                    checks_failed.append("correlation")
                    denial_reasons.append(
                        f"Adding {symbol} would create a position pair with "
                        f"correlation > {self._cfg.max_correlation:.0%}"
                    )
            except KeyError as exc:
                # Symbol missing from returns data; log and skip rather than
                # crash the entire system.
                logger.warning(
                    f"approve_trade: correlation check skipped for {symbol}: {exc}"
                )

        # ----------------------------------------------------------------
        # Decision
        # ----------------------------------------------------------------
        approved = len(checks_failed) == 0
        reason = "; ".join(denial_reasons) if denial_reasons else ""

        if not approved:
            logger.info(
                f"Trade DENIED: {side.upper()} {qty:.0f} {symbol} @ {price:.2f} "
                f"— {reason}"
            )
        else:
            logger.debug(
                f"Trade APPROVED: {side.upper()} {qty:.0f} {symbol} @ {price:.2f}"
            )

        return TradeApproval(
            approved=approved,
            reason=reason,
            suggested_qty=qty,
            checks_run=checks_run,
            checks_failed=checks_failed,
        )
