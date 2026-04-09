#!/usr/bin/env python3
"""
pnl_tracker.py — Automated P&L tracking for trading firm documentation.

Runs at EOD. Queries IBKR for today's executions, computes P&L per trade,
updates a persistent CSV ledger, and sends a daily summary via Discord.

This creates the auditable track record needed for Phase 5 (firm structure).

Run: PYTHONPATH=/opt/atnn/app_src python3 /opt/atnn/scripts/pnl_tracker.py
"""

from __future__ import annotations
import asyncio, csv, json, logging, os, subprocess, sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pnl")

IBKR_PORT  = 4003
CLIENT_ID  = 60
LEDGER_CSV = Path("/opt/atnn/data/trade_ledger.csv")
DAILY_JSON = Path("/opt/atnn/data/daily_pnl.json")
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1482171912724545638/"
    "EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1"
)

EDT = timezone(timedelta(hours=-4))


async def collect_pnl():
    from ib_async import IB

    ib = IB()
    await asyncio.wait_for(ib.connectAsync("127.0.0.1", IBKR_PORT, clientId=CLIENT_ID), 15)

    # Account summary
    acct = await ib.accountSummaryAsync()
    nav = float(next((s.value for s in acct if s.tag == "NetLiquidation"), 0))
    cash = float(next((s.value for s in acct if s.tag == "TotalCashValue"), 0))
    upnl = float(next((s.value for s in acct if s.tag == "UnrealizedPnL"), 0))
    rpnl = float(next((s.value for s in acct if s.tag == "RealizedPnL"), 0))

    # Today's executions
    fills = ib.fills()
    today_str = date.today().isoformat()
    today_fills = []
    for f in fills:
        exec_time = f.execution.time
        if hasattr(exec_time, "date"):
            exec_date = exec_time.date()
        else:
            exec_date = date.today()

        today_fills.append({
            "time": str(exec_time),
            "symbol": f.contract.localSymbol,
            "action": f.execution.side,
            "qty": int(f.execution.shares),
            "price": f.execution.avgPrice,
            "commission": f.commissionReport.commission if f.commissionReport else 0,
            "realized_pnl": f.commissionReport.realizedPNL if f.commissionReport else 0,
        })

    # Current positions
    positions = []
    for p in ib.positions():
        if int(p.position) != 0:
            positions.append({
                "symbol": p.contract.localSymbol,
                "qty": int(p.position),
                "avg_cost": round(p.avgCost / 5, 2),  # MES multiplier
            })

    # Open orders
    open_orders = []
    orders = await ib.reqOpenOrdersAsync()
    for t in orders:
        o = t.order
        px = o.auxPrice if o.orderType in ("STP", "TRAIL") else o.lmtPrice
        open_orders.append({
            "symbol": t.contract.localSymbol,
            "action": o.action,
            "type": o.orderType,
            "qty": int(o.totalQuantity),
            "price": px,
        })

    ib.disconnect()

    # Write to trade ledger CSV (append)
    LEDGER_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LEDGER_CSV.exists()
    with open(LEDGER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "time", "symbol", "action", "qty", "price",
            "commission", "realized_pnl",
        ])
        if write_header:
            w.writeheader()
        for fill in today_fills:
            w.writerow({
                "date": today_str,
                "time": fill["time"],
                "symbol": fill["symbol"],
                "action": fill["action"],
                "qty": fill["qty"],
                "price": fill["price"],
                "commission": round(fill["commission"], 2),
                "realized_pnl": round(fill["realized_pnl"], 2),
            })

    # Daily summary JSON
    daily = {
        "date": today_str,
        "nav": nav,
        "cash": cash,
        "unrealized_pnl": upnl,
        "realized_pnl": rpnl,
        "fills_today": len(today_fills),
        "positions": positions,
        "open_orders": len(open_orders),
    }

    # Load history
    history = []
    if DAILY_JSON.exists():
        try:
            history = json.loads(DAILY_JSON.read_text())
        except Exception:
            history = []

    # Update or append today
    history = [d for d in history if d["date"] != today_str]
    history.append(daily)
    history.sort(key=lambda x: x["date"])
    DAILY_JSON.write_text(json.dumps(history, indent=2, default=str))

    # Compute running stats
    navs = [d["nav"] for d in history]
    if len(navs) >= 2:
        daily_returns = [(navs[i] - navs[i-1]) / navs[i-1] for i in range(1, len(navs))]
        import numpy as np
        total_ret = (navs[-1] - navs[0]) / navs[0]
        avg_daily = np.mean(daily_returns) if daily_returns else 0
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
        max_nav = max(navs)
        drawdown = (navs[-1] - max_nav) / max_nav if max_nav > 0 else 0
    else:
        total_ret = (nav - 5923) / 5923
        sharpe = 0
        drawdown = 0

    # Discord summary
    now_edt = datetime.now(EDT)
    pos_str = ", ".join(f"{p['symbol']}:{p['qty']}@{p['avg_cost']}" for p in positions) or "FLAT"
    fills_str = f"{len(today_fills)} fills" if today_fills else "no fills"

    msg = (
        f"**EOD {now_edt.strftime('%Y-%m-%d')}**\n"
        f"NAV: ${nav:,.2f} | Unrealized: ${upnl:,.2f} | Realized: ${rpnl:,.2f}\n"
        f"Positions: {pos_str}\n"
        f"Today: {fills_str}\n"
        f"Track record: {len(history)} days | Total: {total_ret*100:+.2f}% | "
        f"Sharpe: {sharpe:.2f} | DD: {drawdown*100:.2f}%"
    )

    log.info(msg)

    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: curl/7.68.0",
            "-d", json.dumps({"content": msg}),
            DISCORD_WEBHOOK,
        ], capture_output=True, timeout=10)
    except Exception:
        pass

    return daily


if __name__ == "__main__":
    asyncio.run(collect_pnl())
