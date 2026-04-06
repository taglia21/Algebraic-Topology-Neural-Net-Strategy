"""
core/reconciler.py
==================
Position reconciliation: ensures strategy state matches IBKR truth.

Principle: Broker is ALWAYS ground truth.

Workflow (per research — Headlands Technologies, OpenClaw/Tencent):
  1. Query IBKR positions
  2. Compare to strategy state
  3. If delta: wait 3s (timing grace), re-query
  4. If still different: sync state to broker, halt new entries, alert

Key behaviors:
  - Never automatically "trade out" of a large reconciliation break
  - For delta == 1 contract: sync and log (can happen from manual trades)
  - For delta > 1 contract: sync, halt, alert (something is very wrong)
  - Always run at the START of each cycle, before placing any orders
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PositionBreak:
    symbol: str
    broker_qty: int
    strategy_qty: int
    delta: int


class PositionReconciler:
    """
    Reconciles strategy state vs IBKR positions.
    
    Usage:
        rec = PositionReconciler(ib)
        breaks = await rec.reconcile(strategy_positions)
        if rec.should_halt:
            return  # don't place new orders
    """

    def __init__(self, ib):
        self.ib = ib
        self.should_halt = False
        self.breaks: list[PositionBreak] = []

    async def reconcile(self, strategy_positions: dict[str, int]) -> list[PositionBreak]:
        """
        Compare broker positions to strategy state.
        
        Parameters
        ----------
        strategy_positions : dict
            {symbol: quantity} from strategy state file.
            
        Returns
        -------
        list of PositionBreak objects (empty if all good).
        """
        # Query broker (ground truth)
        broker_positions = {}
        try:
            for pos in self.ib.positions():
                sym = pos.contract.symbol
                qty = int(pos.position)
                if qty != 0:
                    broker_positions[sym] = qty
        except Exception as e:
            log.error("Reconciler: failed to get broker positions: %s", e)
            return []

        # Compare
        all_symbols = set(strategy_positions.keys()) | set(broker_positions.keys())
        breaks = []

        for sym in all_symbols:
            broker_qty   = broker_positions.get(sym, 0)
            strategy_qty = strategy_positions.get(sym, 0)

            if broker_qty != strategy_qty:
                breaks.append(PositionBreak(sym, broker_qty, strategy_qty,
                                            broker_qty - strategy_qty))

        if not breaks:
            log.debug("Reconciler: all positions match broker")
            return []

        # Timing grace: wait 3 seconds and re-check
        log.warning("Reconciler: %d position breaks detected. Waiting 3s...", len(breaks))
        await asyncio.sleep(3.0)

        # Re-query
        refreshed = {}
        try:
            for pos in self.ib.positions():
                sym = pos.contract.symbol
                qty = int(pos.position)
                if qty != 0:
                    refreshed[sym] = qty
        except Exception:
            refreshed = broker_positions

        confirmed_breaks = []
        for brk in breaks:
            fresh_qty = refreshed.get(brk.symbol, 0)
            strat_qty = strategy_positions.get(brk.symbol, 0)
            if fresh_qty != strat_qty:
                confirmed_breaks.append(PositionBreak(
                    brk.symbol, fresh_qty, strat_qty, fresh_qty - strat_qty
                ))
            else:
                log.info("Reconciler: %s resolved itself (timing break)", brk.symbol)

        self.breaks = confirmed_breaks

        for brk in confirmed_breaks:
            log.error(
                "POSITION BREAK CONFIRMED: %s broker=%d strategy=%d delta=%d",
                brk.symbol, brk.broker_qty, brk.strategy_qty, brk.delta,
            )
            # Sync state to broker (broker wins)
            strategy_positions[brk.symbol] = brk.broker_qty

            # Halt on large breaks
            if abs(brk.delta) > 1:
                self.should_halt = True
                log.error("  → Delta > 1: HALTING new entries pending human review")

        return confirmed_breaks

    def get_actual_position(self, symbol: str,
                            strategy_positions: dict[str, int]) -> int:
        """
        Return the actual position size, using broker truth if available.
        Falls back to strategy state.
        """
        # Check if we have a recent break for this symbol
        for brk in self.breaks:
            if brk.symbol == symbol:
                return brk.broker_qty
        return strategy_positions.get(symbol, 0)
