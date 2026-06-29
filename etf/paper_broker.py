"""Local simulated paper broker for ETF engine fail-safe operation.

This adapter mirrors the IBKR broker interface used by :mod:`etf.main` so
paper-mode cycles can continue when IBKR Gateway/TWS is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from etf.config import ExecutionConfig
from etf.ibkr_broker import (
    AccountSnapshot,
    PlannedOrder,
    ReconciliationReport,
    compute_reconciliation,
)
from etf.strategy import enforce_gross_cap

logger = logging.getLogger("etf.paper_broker")


class SimulatedPaperBroker:
    """Drop-in paper broker with persistent local account state."""

    def __init__(self, execution_cfg: ExecutionConfig, *, dry_run: bool = True) -> None:
        self.execution_cfg = execution_cfg
        self.dry_run = dry_run
        self._connected = False
        self._state_path = Path(
            os.environ.get("ETF_PAPER_SIM_STATE_PATH", ".etf_telemetry/paper_sim_account.json")
        )
        self._initial_equity = float(os.environ.get("ETF_PAPER_SIM_INITIAL_EQUITY", "100000"))

        self._cash = self._initial_equity
        self._positions: Dict[str, float] = {}
        self._last_equity = self._initial_equity
        self._last_prices: Dict[str, float] = {}

        self.last_orders: List[PlannedOrder] = []
        self.last_fills: Dict[str, float] = {}

    async def connect(self) -> bool:
        self._load_state()
        self._connected = True
        logger.warning("Using local simulated paper broker fallback (IBKR unavailable).")
        return True

    async def disconnect(self) -> None:
        self._save_state()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text())
            self._cash = float(raw.get("cash", self._initial_equity))
            self._positions = {str(k): float(v) for k, v in dict(raw.get("positions", {})).items()}
            self._last_equity = float(raw.get("last_equity", self._initial_equity))
        except Exception as exc:
            logger.error("Failed to load paper-sim state %s: %s", self._state_path, exc)

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cash": self._cash,
                "positions": self._positions,
                "last_equity": self._last_equity,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._state_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.error("Failed to save paper-sim state %s: %s", self._state_path, exc)

    def _mark_to_market_equity(self) -> float:
        eq = float(self._cash)
        for sym, shares in self._positions.items():
            px = self._last_prices.get(sym)
            if px is not None and px > 0:
                eq += float(shares) * float(px)
        return eq

    async def get_account(self) -> Optional[AccountSnapshot]:
        equity = self._mark_to_market_equity()
        if equity <= 0:
            equity = self._last_equity
        self._last_equity = float(equity)
        return AccountSnapshot(
            equity=float(equity),
            cash=float(self._cash),
            buying_power=float(self._cash),
            positions={k: float(v) for k, v in self._positions.items()},
        )

    async def plan_rebalance(
        self,
        target_weights: Dict[str, float],
        cfg,
        fallback_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[List[PlannedOrder]]:
        account = await self.get_account()
        if account is None or account.equity <= 0:
            return None

        equity = float(account.equity)
        symbols = sorted(set(target_weights) | set(account.positions))
        prices: Dict[str, float] = {}
        for sym in symbols:
            px = float((fallback_prices or {}).get(sym, 0.0) or 0.0)
            if px <= 0:
                logger.error("Paper-sim missing fallback price for %s; aborting plan.", sym)
                return None
            prices[sym] = px

        self._last_prices = prices
        min_delta_notional = cfg.execution.min_rebalance_delta * equity
        # 1. Effective post-min-delta book: adopt the target where the move
        #    clears the churn threshold, else retain the current weight.
        effective_w: Dict[str, float] = {}
        for sym in symbols:
            tgt_w = float(target_weights.get(sym, 0.0))
            cur_shares = float(account.positions.get(sym, 0.0))
            delta_notional = tgt_w * equity - cur_shares * prices[sym]
            cur_w = (cur_shares * prices[sym]) / equity if equity else 0.0
            effective_w[sym] = (
                tgt_w if abs(delta_notional) >= min_delta_notional else cur_w
            )
        # 2. Strictly enforce the gross-leverage cap on the book about to be held
        #    (mirrors the live IBKR broker and the backtester). No-op within cap.
        effective_w = enforce_gross_cap(effective_w, cfg.risk.max_gross_leverage)
        # 3. Generate orders from current shares -> effective weights.
        orders: List[PlannedOrder] = []
        for sym in symbols:
            eff_w = float(effective_w.get(sym, 0.0))
            cur_shares = float(account.positions.get(sym, 0.0))
            cur_w = (cur_shares * prices[sym]) / equity if equity else 0.0
            delta_notional = eff_w * equity - cur_shares * prices[sym]
            qty = int(round(delta_notional / prices[sym]))
            if qty == 0:
                continue
            action = "BUY" if qty > 0 else "SELL"
            orders.append(
                PlannedOrder(
                    symbol=sym,
                    action=action,
                    quantity=abs(qty),
                    target_weight=eff_w,
                    current_weight=cur_w,
                    est_price=float(prices[sym]),
                    est_notional=abs(qty) * float(prices[sym]),
                    order_type=cfg.execution.order_type.upper(),
                    limit_price=None,
                )
            )
        return orders

    async def execute_orders(self, orders: List[PlannedOrder]) -> Dict[str, str]:
        self.last_orders = list(orders)
        self.last_fills = {}

        if self.dry_run:
            return {o.symbol: "dry_run" for o in orders}

        results: Dict[str, str] = {}
        slip = max(0.0, float(self.execution_cfg.slippage_bps)) / 1e4
        for o in orders:
            sign = 1.0 if o.action == "BUY" else -1.0
            fill_price = o.est_price * (1.0 + slip if o.action == "BUY" else 1.0 - slip)
            qty_signed = sign * float(o.quantity)
            notional = fill_price * qty_signed
            self._positions[o.symbol] = float(self._positions.get(o.symbol, 0.0) + qty_signed)
            if abs(self._positions[o.symbol]) < 1e-9:
                self._positions.pop(o.symbol, None)
            self._cash -= notional
            self.last_fills[o.symbol] = float(fill_price)
            results[o.symbol] = "filled_sim"

        self._last_equity = self._mark_to_market_equity()
        self._save_state()
        return results

    async def collect_fills(self) -> Dict[str, float]:
        return dict(self.last_fills)

    async def await_fills(self, timeout: float = 30.0, poll: float = 0.5) -> bool:
        return True

    async def rebalance_to_weights(
        self,
        target_weights: Dict[str, float],
        cfg,
        fallback_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        orders = await self.plan_rebalance(target_weights, cfg, fallback_prices)
        if orders is None:
            return {"_status": "aborted_failsafe"}
        if not orders:
            return {"_status": "no_change"}
        return await self.execute_orders(orders)

    async def reconcile(
        self,
        target_weights: Dict[str, float],
        cfg,
        fallback_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[ReconciliationReport]:
        account = await self.get_account()
        if account is None or account.equity <= 0:
            return None
        prices: Dict[str, float] = {}
        symbols = sorted(set(target_weights) | set(account.positions))
        for sym in symbols:
            px = float((fallback_prices or {}).get(sym, self._last_prices.get(sym, 0.0)) or 0.0)
            if px <= 0:
                return None
            prices[sym] = px
        self._last_prices = prices
        return compute_reconciliation(
            target_weights,
            account.positions,
            prices,
            account.equity,
            rel_tolerance=cfg.execution.reconciliation_tolerance,
            as_of=datetime.now(timezone.utc).isoformat(),
        )
