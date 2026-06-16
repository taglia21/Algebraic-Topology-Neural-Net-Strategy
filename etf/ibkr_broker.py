"""
etf/ibkr_broker.py
==================
Interactive Brokers execution adapter for the ETF engine via ib_async.

Trades **ETFs only** as `Stock` contracts (SMART routing, USD). The core
primitive is :meth:`rebalance_to_weights` — given a dict of target portfolio
weights, it reads the live account, computes share deltas, and submits the
orders needed to move the book to target. This is intentionally distinct from
the VRP engine's options-combo logic.

Safety features
---------------
- Read-only / dry-run mode (compute and log orders without submitting).
- Per-order notional sanity checks and a configurable max single-order cap.
- Minimum-rebalance threshold to avoid churning on tiny weight drifts.
- Fail-safe: any data/connection problem aborts the rebalance (no blind orders).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from etf.config import ETFConfig, IBKRConfig

logger = logging.getLogger("etf.ibkr")


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    positions: Dict[str, float]  # symbol -> share quantity (signed)


@dataclass
class PlannedOrder:
    symbol: str
    action: str          # "BUY" / "SELL"
    quantity: int        # whole shares
    target_weight: float
    current_weight: float
    est_price: float
    est_notional: float
    order_type: str = "MKT"          # "MKT" or "LMT"
    limit_price: Optional[float] = None  # marketable-limit cap when order_type == "LMT"


@dataclass
class ReconciliationReport:
    """Post-trade reconciliation of realised vs target weights (ETF-only)."""

    ok: bool
    as_of: str
    equity: float
    mismatches: Dict[str, float]   # symbol -> |realised_w - target_w| beyond tolerance
    realised_weights: Dict[str, float]
    target_weights: Dict[str, float]


def _finite(x) -> Optional[float]:
    """Return ``x`` as a positive float if it is finite and > 0, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v <= 0:  # NaN or non-positive
        return None
    return v


# Terminal IBKR order states — once an order reaches one of these it will not
# fill further, so the fill-wait loop can stop polling it. "PendingSubmit",
# "PreSubmitted", "Submitted" and "" are still *working*.
_TERMINAL_ORDER_STATES = frozenset({
    "Filled", "Cancelled", "ApiCancelled", "Inactive",
})


def _order_is_done(status: str) -> bool:
    """True if an IBKR order status is terminal (will not fill further)."""
    return (status or "").strip() in _TERMINAL_ORDER_STATES


def _extract_price(ticker) -> Optional[float]:
    """Pick a usable price from an ib_async Ticker, handling real-time AND
    delayed data. Preference order: mid (bid/ask) -> last -> close, trying the
    real-time field then its ``delayed*`` counterpart. Pure/defensive: returns
    None if nothing usable is present (so the caller aborts fail-safe).
    """
    if ticker is None:
        return None
    bid = _finite(getattr(ticker, "bid", None)) or _finite(getattr(ticker, "delayedBid", None))
    ask = _finite(getattr(ticker, "ask", None)) or _finite(getattr(ticker, "delayedAsk", None))
    if bid and ask:
        return (bid + ask) / 2.0
    last = _finite(getattr(ticker, "last", None)) or _finite(getattr(ticker, "delayedLast", None))
    if last:
        return last
    close = _finite(getattr(ticker, "close", None)) or _finite(getattr(ticker, "delayedClose", None))
    return close


def marketable_limit_price(action: str, ref_price: float, offset_bps: float) -> float:
    """Marketable-limit price: cross the spread by ``offset_bps`` to get a fast
    fill while capping slippage.

    BUY  -> ref * (1 + offset)  (willing to pay slightly up, but no more)
    SELL -> ref * (1 - offset)  (willing to receive slightly less, but no less)

    A marketable limit behaves like a market order in normal liquidity yet
    protects against a blown-out spread or a stale quote — exactly the execution
    realism the Phase 5 gate requires. Rounded to the penny (US ETF tick).
    """
    if ref_price <= 0:
        raise ValueError("ref_price must be positive")
    off = max(0.0, offset_bps) / 1e4
    raw = ref_price * (1.0 + off) if action.upper() == "BUY" else ref_price * (1.0 - off)
    return round(max(0.01, raw), 2)


def compute_reconciliation(
    target_weights: Dict[str, float],
    positions: Dict[str, float],
    prices: Dict[str, float],
    equity: float,
    *,
    tolerance: float = 0.02,
    as_of: str = "",
) -> ReconciliationReport:
    """Compare realised portfolio weights to target; flag any |delta| > tolerance.

    Pure function (no broker dependency) so it is fully unit-testable. ``equity``
    and ``prices`` come from the broker snapshot; ``positions`` is symbol->shares.
    A mismatch means the live book drifted from intent (partial fill, rejected
    order, stale data) and must be investigated before the next cycle — the
    Paper→Live gate requires zero unresolved mismatches.
    """
    realised: Dict[str, float] = {}
    if equity > 0:
        for sym, shares in positions.items():
            px = prices.get(sym)
            if px and px > 0:
                realised[sym] = (shares * px) / equity
    symbols = set(target_weights) | set(realised)
    mismatches: Dict[str, float] = {}
    for sym in symbols:
        delta = abs(realised.get(sym, 0.0) - target_weights.get(sym, 0.0))
        if delta > tolerance:
            mismatches[sym] = float(delta)
    return ReconciliationReport(
        ok=len(mismatches) == 0,
        as_of=as_of,
        equity=float(equity),
        mismatches=mismatches,
        realised_weights={k: float(v) for k, v in realised.items()},
        target_weights={k: float(v) for k, v in target_weights.items()},
    )


class IBKRETFBroker:
    """ETF execution adapter. Long-only by default (matches the strategy)."""

    def __init__(self, config: IBKRConfig, *, dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run
        self._ib = None
        self._connected = False
        # Slippage-telemetry capture: the orders and realised fills from the
        # most recent execute_orders call (live only; dry-run leaves fills empty).
        self.last_orders: List[PlannedOrder] = []
        self.last_fills: Dict[str, float] = {}
        self._last_trades: Dict[str, object] = {}

    # -- connection -------------------------------------------------------
    async def connect(self) -> bool:
        try:
            try:
                from ib_async import IB
                logger.info("Using ib_async library")
            except ImportError:
                from ib_insync import IB
                logger.info("Using ib_insync library (fallback)")

            self._ib = IB()
            await self._ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly or self.dry_run,
            )
            self._connected = True
            # Select the market-data type up-front. Paper accounts rarely carry
            # a real-time subscription, so default to delayed (type 3) to keep
            # quotes flowing (NaN real-time fields would otherwise abort the
            # rebalance fail-safe). Best-effort: never fail the connection on it.
            try:
                self._ib.reqMarketDataType(self.config.market_data_type)
                logger.info("Market-data type set to %s", self.config.market_data_type)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not set market-data type: %s", exc)
            logger.info(
                "Connected to IBKR %s:%s (client %s, dry_run=%s)",
                self.config.host, self.config.port, self.config.client_id, self.dry_run,
            )
            return True
        except Exception as exc:
            logger.error("IBKR connection failed: %s", exc)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    # -- contracts & data -------------------------------------------------
    def _stock(self, symbol: str):
        try:
            from ib_async import Stock
        except ImportError:
            from ib_insync import Stock
        return Stock(symbol, "SMART", "USD")

    async def _qualify(self, symbol: str):
        contract = self._stock(symbol)
        await self._ib.qualifyContractsAsync(contract)
        return contract

    async def get_price(self, symbol: str, price_timeout: float = 10.0) -> Optional[float]:
        """Last/mid price for an ETF. Fail-safe returns None on any issue.

        Handles both real-time and delayed market data: under delayed data
        (``market_data_type`` 3/4) IBKR populates ``delayedLast``/``delayedBid``/
        ``delayedAsk``/``delayedClose`` instead of the real-time fields, so we
        probe both. Mid (bid/ask) is preferred over last for a tighter mark.

        Polls for a valid quote up to ``price_timeout`` seconds rather than using
        a single fixed wait: the FIRST snapshot after a quiet period (especially
        under delayed data) can take several seconds to populate, so a fixed
        short sleep intermittently yields ``None`` and trips the rebalance
        fail-safe. Returning as soon as a price appears keeps latency low while
        tolerating slow first ticks.
        """
        if not self.is_connected:
            return None
        import time as _time
        contract = None
        try:
            contract = await self._qualify(symbol)
            ticker = self._ib.reqMktData(contract, "", False, False)
            price = None
            deadline = _time.monotonic() + max(2.0, price_timeout)
            while _time.monotonic() < deadline:
                # MUST use asyncio.sleep here, NOT ib.sleep(): ib.sleep() calls
                # util.run() -> loop.run_until_complete() which re-enters the
                # already-running event loop ("This event loop is already
                # running") and aborts the fetch. awaiting asyncio.sleep yields
                # to the loop, which processes the incoming IBKR ticks and
                # populates the ticker in the background.
                await asyncio.sleep(0.25)
                price = _extract_price(ticker)
                if price is not None and price > 0:
                    break
            if price is None or price <= 0:
                logger.warning(
                    "No price for %s after %.1fs (market-data slow or no "
                    "subscription/permission).", symbol, price_timeout,
                )
            return price
        except Exception as exc:
            logger.error("Price fetch failed for %s: %s", symbol, exc)
            return None
        finally:
            if contract is not None:
                try:
                    self._ib.cancelMktData(contract)
                except Exception:  # pragma: no cover - defensive
                    pass

    async def get_account(self) -> Optional[AccountSnapshot]:
        if not self.is_connected:
            return None
        try:
            # Use the high-level ASYNC variants to avoid re-entering the running
            # event loop. NOTE: ib.reqAccountSummaryAsync() only *starts* the
            # subscription and resolves to an empty list — the parsed rows live in
            # ib_async's cache. ib.accountSummaryAsync() both triggers the request
            # (on demand) AND returns the parsed list, so we use that here.
            summary_rows = await self._ib.accountSummaryAsync()
            summary = {row.tag: row.value for row in summary_rows}
            equity = float(summary.get("NetLiquidation", 0.0) or 0.0)
            cash = float(summary.get("TotalCashValue", 0.0) or 0.0)
            bp = float(summary.get("BuyingPower", 0.0) or 0.0)

            # Fallback: if the summary is empty (timing/edge cases), derive the
            # core balances from the account-update stream IBKR pushes during the
            # initial sync on connect. Only consider USD/base-currency rows.
            if equity <= 0:
                for v in self._ib.accountValues():
                    if v.currency not in ("", "USD", "BASE"):
                        continue
                    if v.tag == "NetLiquidation":
                        equity = float(v.value or 0.0)
                    elif v.tag == "TotalCashValue":
                        cash = float(v.value or 0.0)
                    elif v.tag == "BuyingPower":
                        bp = float(v.value or 0.0)

            positions: Dict[str, float] = {}
            for p in await self._ib.reqPositionsAsync():
                sym = getattr(p.contract, "symbol", None)
                if sym:
                    positions[sym] = positions.get(sym, 0.0) + float(p.position)
            return AccountSnapshot(equity=equity, cash=cash, buying_power=bp, positions=positions)
        except Exception as exc:
            logger.error("Account fetch failed: %s", exc)
            return None

    # -- rebalancing ------------------------------------------------------
    async def plan_rebalance(
        self,
        target_weights: Dict[str, float],
        cfg: ETFConfig,
    ) -> Optional[List[PlannedOrder]]:
        """Compute the orders required to reach ``target_weights``.

        Returns None on any failure (fail-safe: caller must not trade blindly).
        """
        account = await self.get_account()
        if account is None or account.equity <= 0:
            logger.error("No valid account snapshot; aborting rebalance plan.")
            return None

        equity = account.equity
        # Universe to consider = targets plus anything currently held (to exit).
        symbols = sorted(set(target_weights) | set(account.positions))

        prices: Dict[str, float] = {}
        for sym in symbols:
            px = await self.get_price(sym)
            if px is None or px <= 0:
                logger.error("Missing price for %s; aborting rebalance (fail-safe).", sym)
                return None
            prices[sym] = px

        min_delta_notional = cfg.execution.min_rebalance_delta * equity
        orders: List[PlannedOrder] = []
        for sym in symbols:
            tgt_w = target_weights.get(sym, 0.0)
            cur_shares = account.positions.get(sym, 0.0)
            cur_w = (cur_shares * prices[sym]) / equity if equity else 0.0
            target_notional = tgt_w * equity
            delta_notional = target_notional - cur_shares * prices[sym]
            if abs(delta_notional) < min_delta_notional:
                continue  # below churn threshold; skip
            qty = int(round(delta_notional / prices[sym]))
            if qty == 0:
                continue
            action = "BUY" if qty > 0 else "SELL"
            otype = cfg.execution.order_type.upper()
            lpx = (
                marketable_limit_price(action, prices[sym], cfg.execution.limit_offset_bps)
                if otype == "LMT" else None
            )
            orders.append(PlannedOrder(
                symbol=sym,
                action=action,
                quantity=abs(qty),
                target_weight=tgt_w,
                current_weight=cur_w,
                est_price=prices[sym],
                est_notional=abs(qty) * prices[sym],
                order_type=otype,
                limit_price=lpx,
            ))
        return orders

    async def execute_orders(self, orders: List[PlannedOrder]) -> Dict[str, str]:
        """Submit planned orders. In dry-run mode, only logs them."""
        results: Dict[str, str] = {}
        # Reset telemetry capture for this cycle.
        self.last_orders = list(orders)
        self.last_fills = {}
        self._last_trades = {}
        if self.dry_run:
            for o in orders:
                logger.info(
                    "[DRY-RUN] %s %d %s @ ~%.2f (%.1f%% notional)",
                    o.action, o.quantity, o.symbol, o.est_price,
                    100 * o.est_notional,
                )
                results[o.symbol] = "dry_run"
            return results

        if not self.is_connected:
            logger.error("Not connected; cannot execute orders.")
            return {o.symbol: "no_connection" for o in orders}

        try:
            from ib_async import MarketOrder, LimitOrder
        except ImportError:
            from ib_insync import MarketOrder, LimitOrder

        for o in orders:
            try:
                contract = await self._qualify(o.symbol)
                if self.config.readonly:
                    results[o.symbol] = "readonly_blocked"
                    continue
                if o.order_type == "LMT" and o.limit_price:
                    order = LimitOrder(o.action, o.quantity, o.limit_price)
                    logger.info("Submitted %s %d %s LMT @ %.2f",
                                o.action, o.quantity, o.symbol, o.limit_price)
                else:
                    order = MarketOrder(o.action, o.quantity)
                    logger.info("Submitted %s %d %s MKT", o.action, o.quantity, o.symbol)
                trade = self._ib.placeOrder(contract, order)
                self._last_trades[o.symbol] = trade
                results[o.symbol] = "submitted"
            except Exception as exc:
                logger.error("Order failed for %s: %s", o.symbol, exc)
                results[o.symbol] = f"error:{exc}"
        return results

    async def collect_fills(self) -> Dict[str, float]:
        """Best-effort realised average fill price per symbol from the last
        execute_orders cycle, for slippage telemetry. Populates ``last_fills``.

        Reads ``trade.orderStatus.avgFillPrice`` once orders have filled. Returns
        an empty dict in dry-run or if no fills are available (fail-safe — never
        raises so telemetry can't break the trading loop).
        """
        fills: Dict[str, float] = {}
        for sym, trade in self._last_trades.items():
            try:
                px = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
                if px > 0:
                    fills[sym] = px
            except Exception:  # pragma: no cover - defensive
                continue
        self.last_fills = fills
        return fills

    async def await_fills(self, timeout: float = 30.0, poll: float = 0.5) -> bool:
        """Block until every submitted order reaches a terminal state, or until
        ``timeout`` seconds elapse — whichever comes first.

        Returns True if all orders settled (Filled/Cancelled/etc.) within the
        window, False if it timed out with at least one still working. Reading
        positions/fills BEFORE orders settle yields a spurious reconciliation
        mismatch that would block the next cycle, so the trade loop calls this
        between submit and reconcile. No-op (returns True) in dry-run or when no
        live trades were placed. Fail-safe: never raises.
        """
        if self.dry_run or not self._last_trades:
            return True
        import time as _time
        deadline = _time.monotonic() + max(0.0, timeout)
        while True:
            statuses = []
            for trade in self._last_trades.values():
                try:
                    statuses.append(str(getattr(trade.orderStatus, "status", "") or ""))
                except Exception:  # pragma: no cover - defensive
                    statuses.append("")
            if all(_order_is_done(s) for s in statuses):
                return True
            if _time.monotonic() >= deadline:
                working = [s for s in statuses if not _order_is_done(s)]
                logger.warning(
                    "await_fills timed out after %.1fs; %d order(s) still working: %s",
                    timeout, len(working), working,
                )
                return False
            # asyncio.sleep (NOT ib.sleep) — yields to the running loop so ib_async
            # processes order-status updates without re-entering the loop.
            await asyncio.sleep(poll)

    async def rebalance_to_weights(
        self, target_weights: Dict[str, float], cfg: ETFConfig
    ) -> Dict[str, str]:
        """End-to-end: plan + execute a rebalance to the target weights."""
        orders = await self.plan_rebalance(target_weights, cfg)
        if orders is None:
            return {"_status": "aborted_failsafe"}
        if not orders:
            logger.info("Portfolio already within tolerance; no orders.")
            return {"_status": "no_change"}
        return await self.execute_orders(orders)

    async def reconcile(
        self, target_weights: Dict[str, float], cfg: ETFConfig
    ) -> Optional[ReconciliationReport]:
        """Post-trade check: do realised broker weights match the target book?

        Returns None (fail-safe) if the account or any price is unavailable.
        Any mismatch beyond the rebalance threshold is flagged for the runbook;
        the Paper→Live gate requires zero unresolved mismatches.
        """
        account = await self.get_account()
        if account is None or account.equity <= 0:
            logger.error("Cannot reconcile: no valid account snapshot.")
            return None
        symbols = sorted(set(target_weights) | set(account.positions))
        prices: Dict[str, float] = {}
        for sym in symbols:
            px = await self.get_price(sym)
            if px is None or px <= 0:
                logger.error("Cannot reconcile: missing price for %s.", sym)
                return None
            prices[sym] = px
        report = compute_reconciliation(
            target_weights,
            account.positions,
            prices,
            account.equity,
            tolerance=cfg.execution.min_rebalance_delta,
        )
        if report.ok:
            logger.info("Reconciliation OK (%d symbols, equity %.2f).",
                        len(symbols), account.equity)
        else:
            logger.warning("Reconciliation MISMATCH: %s", report.mismatches)
        return report
