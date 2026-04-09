#!/usr/bin/env python3
"""
intraday_scanner.py — High-Frequency TDA Trading on 5-Min MES Bars
====================================================================
Runs every 30 minutes during market hours. Pulls 5-min MES data directly
from IBKR, computes TDA on the rolling intraday window, and trades
regime shifts in real-time.

This bridges the gap between our daily system (~20 trades/year) and
Joshua's ORIA (~3,798 trades/year) by operating on 5-minute bars.

Schedule: */30 9-16 * * 1-5 (every 30 min, 9:00 AM - 4:00 PM ET)

Signal logic:
  - Compute TDA on last 40 bars of 5-min MES data (= 3.3 hour window)
  - spectral_gap drops below p25 + momentum positive → LONG 1 MES
  - spectral_gap rises above p75 or wasserstein spikes → EXIT
  - Tighter intraday stops: 1.5× intraday ATR
  - Target: 2× stop distance (2:1 R:R)

Position management:
  - Max 1 intraday MES contract (separate from daily regime position)
  - Flatten at 3:55 PM if still holding (no overnight intraday risk)
  - Circuit breaker: halt if daily intraday P&L < -1% of NAV
"""

from __future__ import annotations
import asyncio, json, logging, os, sys, tempfile
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

APP_SRC = Path('/opt/atnn/app_src')
sys.path.insert(0, str(APP_SRC))

from tda.extractor import TDAFeatureExtractor
from nn.regime_labeler import heuristic_regime, regime_name, HEURISTIC_THRESHOLDS
from tda.composite_scorer import TDACompositeScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("intraday")

# ─── Config ───────────────────────────────────────────────────────────────────

IBKR_HOST   = "127.0.0.1"
IBKR_PORT   = 4003
CLIENT_ID   = 45          # separate from daily system (clientId=40)

MES_SYMBOL  = "MES"
MES_EXCHANGE = "CME"

MAX_INTRADAY_CONTRACTS = 1
INTRADAY_ATR_STOP_MULT = 1.5    # tighter than daily (2.0)
INTRADAY_TARGET_MULT   = 2.0    # 2:1 R:R
INTRADAY_LOSS_LIMIT    = 0.01   # 1% daily intraday loss limit
FLATTEN_TIME           = dtime(15, 55)  # 3:55 PM ET — flatten before close

TDA_WINDOW = 40   # 40 bars × 5 min = 200 minutes = 3.3 hours
TDA_MIN_BARS = 30  # need at least 30 bars to compute meaningful TDA

STATE_FILE = Path("/opt/atnn/data/intraday_state.json")

DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1482171912724545638/"
    "EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1"
)

# Intraday-calibrated thresholds (will be updated from live data)
# These are initial estimates; the scanner self-calibrates from recent bars
INTRADAY_THRESHOLDS = {
    "spec_gap_trending_below": None,   # computed from live data percentile
    "spec_gap_reverting_above": None,
    "wass_volatile_above": None,
    "mom_positive": 0.0005,            # 0.05% per 5-min bar = trending
}


# ─── State ────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "date": str(date.today()),
        "scans_today": 0,
        "intraday_trades": 0,
        "intraday_pnl": 0.0,
        "position": 0,
        "entry_price": 0.0,
        "halted": False,
        "regime_history": [],
    }


def save_state(s: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(s, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def discord(msg: str):
    import subprocess
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: curl/7.68.0",
            "-d", json.dumps({"content": f"**ATNN Intraday**: {msg}"}),
            DISCORD_WEBHOOK,
        ], capture_output=True, timeout=10)
    except Exception:
        pass


# ─── MES Contract ─────────────────────────────────────────────────────────────

def get_front_month() -> str:
    today = date.today()
    def third_friday(year, month):
        first = date(year, month, 1)
        days_to_friday = (4 - first.weekday()) % 7
        return first + timedelta(days=days_to_friday) + timedelta(weeks=2)
    for year in [today.year, today.year + 1]:
        for month in [3, 6, 9, 12]:
            if today <= third_friday(year, month) - timedelta(days=8):
                return f"{year}{month:02d}"
    return f"{today.year + 1}03"


async def get_contract(ib):
    from ib_async import Future
    contract = Future(symbol=MES_SYMBOL, lastTradeDateOrContractMonth=get_front_month(),
                      exchange=MES_EXCHANGE, currency="USD", multiplier="5")
    qualified = await ib.qualifyContractsAsync(contract)
    return qualified[0] if qualified else None


def tick_round(price: float) -> float:
    """Round to MES 0.25 tick size."""
    return round(price * 4) / 4


# ─── Intraday TDA Signal ─────────────────────────────────────────────────────

async def get_5min_bars(ib, contract, duration: str = "1 D") -> pd.DataFrame:
    """Pull 5-minute bars from IBKR."""
    bars = await ib.reqHistoricalDataAsync(
        contract, endDateTime="", durationStr=duration,
        barSizeSetting="5 mins", whatToShow="TRADES",
        useRTH=True, formatDate=1
    )
    if not bars:
        return pd.DataFrame()

    records = [{"date": b.date, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": b.volume}
               for b in bars]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def compute_intraday_signal(bars: pd.DataFrame) -> dict:
    """
    Compute TDA regime on 5-min MES bars.

    Returns dict with regime, confidence, and TDA features.
    """
    if len(bars) < TDA_MIN_BARS:
        return {"regime": 2, "confidence": 0.40, "action": "WAIT",
                "reason": f"Only {len(bars)} bars (need {TDA_MIN_BARS})"}

    close = bars["close"]
    high  = bars["high"]
    low   = bars["low"]

    # Compute TDA on the rolling window
    ext = TDAFeatureExtractor(window=min(TDA_WINDOW, len(close) - 5), stride=1)
    tda = ext.extract_series(close)

    if len(tda) < 3:
        return {"regime": 2, "confidence": 0.40, "action": "WAIT",
                "reason": "Insufficient TDA features"}

    last = tda.iloc[-1]
    sg  = float(last.get("spectral_gap", 0.5))
    b1  = float(last.get("beta_1", 0))
    wd  = float(last.get("wasserstein_dist", 0))
    pe  = float(last.get("persistence_entropy", 0))

    # Self-calibrate thresholds from the intraday distribution
    sg_p25 = float(tda["spectral_gap"].quantile(0.25))
    sg_p75 = float(tda["spectral_gap"].quantile(0.75))
    wd_p90 = float(tda["wasserstein_dist"].quantile(0.90))

    # Intraday momentum (5-bar = 25 minutes)
    mom_5bar = float(close.pct_change(5).iloc[-1]) if len(close) >= 5 else 0
    # ATR for stop calculation (14-bar intraday ATR)
    atr = float((high - low).rolling(14).mean().iloc[-1]) if len(bars) >= 14 else 2.0

    # Regime classification using self-calibrated thresholds
    if wd > wd_p90 and wd_p90 > 0:
        regime, conf = 3, 0.62   # VOLATILE — regime transition
        action = "EXIT" if True else "WAIT"
    elif sg < sg_p25 and mom_5bar > 0.0003:
        regime, conf = 0, 0.60   # TRENDING UP
        action = "BUY"
    elif sg < sg_p25 and mom_5bar < -0.0003:
        regime, conf = 1, 0.60   # TRENDING DOWN
        action = "EXIT"
    elif sg > sg_p75:
        regime, conf = 2, 0.55   # MEAN REVERTING
        action = "WAIT"          # wait for IBS or reversal
    else:
        regime, conf = 2, 0.45   # NEUTRAL
        action = "HOLD" if True else "WAIT"

    # Also compute composite score for higher-quality signal
    try:
        scorer = TDACompositeScorer(train_window=min(60, len(close)-10), forward_bars=3)
        composite = scorer.score_live(tda, close)
        if composite >= 0.60:
            action = "BUY"
            regime = 0
            conf = composite
        elif composite <= 0.35:
            action = "EXIT"
            regime = 1
            conf = 1.0 - composite
        else:
            action = "HOLD" if regime == 0 else "WAIT"
    except Exception as e:
        composite = 0.5

    return {
        "regime": regime,
        "confidence": conf,
        "action": action,
        "sg": sg, "b1": b1, "wd": wd, "pe": pe,
        "sg_p25": sg_p25, "sg_p75": sg_p75, "wd_p90": wd_p90,
        "mom_5bar": mom_5bar,
        "atr": atr,
        "current_price": float(close.iloc[-1]),
        "composite_score": composite,
        "reason": f"composite={composite:.3f} sg={sg:.4f} mom={mom_5bar:.5f}",
    }


# ─── Execution ────────────────────────────────────────────────────────────────

async def place_intraday_bracket(ib, contract, qty: int, entry_price: float,
                                  atr: float) -> bool:
    """Place bracket order with intraday-tight stops."""
    from ib_async import MarketOrder, StopOrder, LimitOrder

    stop_pts  = INTRADAY_ATR_STOP_MULT * atr  # 1.5× intraday ATR
    target_pts = INTRADAY_TARGET_MULT * stop_pts  # 2:1 R:R

    stop_price   = tick_round(entry_price - stop_pts)
    target_price = tick_round(entry_price + target_pts)

    entry_id  = ib.client.getReqId()
    stop_id   = ib.client.getReqId()
    target_id = ib.client.getReqId()

    parent = MarketOrder("BUY", qty)
    parent.orderId = entry_id
    parent.transmit = False

    stop = StopOrder("SELL", qty, stop_price)
    stop.orderId  = stop_id
    stop.parentId = entry_id
    stop.transmit = False

    tp = LimitOrder("SELL", qty, target_price)
    tp.orderId  = target_id
    tp.parentId = entry_id
    tp.transmit = True  # triggers all

    for order in [parent, stop, tp]:
        ib.placeOrder(contract, order)

    log.info("Intraday bracket: BUY %d MES @ mkt | stop=%.2f (%.1f pts) | target=%.2f (%.1f pts)",
             qty, stop_price, stop_pts, target_price, target_pts)

    # Wait for fill
    for _ in range(30):
        await asyncio.sleep(0.5)
        trades = [t for t in ib.trades() if t.order.orderId == entry_id]
        if trades and trades[0].isDone():
            fill = trades[0].orderStatus.avgFillPrice
            log.info("Intraday entry filled @ %.2f", fill)
            return True

    log.warning("Intraday bracket timeout")
    return False


async def close_intraday(ib, contract, qty: int, reason: str = "") -> float:
    """Market-close an intraday position. Returns fill price."""
    from ib_async import MarketOrder

    # Cancel only INTRADAY orders (clientId=45). Do NOT reqGlobalCancel
    # because that kills the daily system's stop-loss orders too.
    for trade in ib.trades():
        if not trade.isDone() and trade.order.clientId == CLIENT_ID:
            ib.cancelOrder(trade.order)
    await asyncio.sleep(0.5)

    action = "SELL" if qty > 0 else "BUY"
    order = MarketOrder(action, abs(qty))
    trade = ib.placeOrder(contract, order)
    log.info("Closing intraday: %s %d MES (%s)", action, abs(qty), reason)

    for _ in range(30):
        await asyncio.sleep(0.5)
        if trade.isDone():
            fill = trade.orderStatus.avgFillPrice
            log.info("Closed @ %.2f", fill)
            return fill

    return 0.0


# ─── Main Scan ────────────────────────────────────────────────────────────────

async def run_scan(dry_run: bool = False):
    state = load_state()

    # Reset state on new day
    if state.get("date") != str(date.today()):
        state = {
            "date": str(date.today()),
            "scans_today": 0,
            "intraday_trades": 0,
            "intraday_pnl": 0.0,
            "position": 0,
            "entry_price": 0.0,
            "halted": False,
            "regime_history": [],
        }

    state["scans_today"] += 1
    now = datetime.now()
    log.info("═══ Intraday Scan #%d — %s ═══", state["scans_today"],
             now.strftime("%H:%M"))

    # Check intraday loss limit
    if state["halted"]:
        log.warning("Intraday HALTED (daily loss limit). Skipping.")
        save_state(state)
        return

    if state["intraday_pnl"] < -(5923 * INTRADAY_LOSS_LIMIT):
        state["halted"] = True
        log.error("Intraday loss limit hit: $%.2f. Halting.", state["intraday_pnl"])
        save_state(state)
        return

    # Connect to IBKR
    from ib_async import IB
    ib = IB()
    try:
        await asyncio.wait_for(ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID), 15)
    except Exception as e:
        log.error("Connect failed: %s", e)
        save_state(state)
        return

    try:
        contract = await get_contract(ib)
        if not contract:
            log.error("Cannot qualify MES contract")
            ib.disconnect()
            save_state(state)
            return

        # Get 5-min bars
        bars = await get_5min_bars(ib, contract, "1 D")
        if len(bars) < TDA_MIN_BARS:
            log.info("Only %d bars available (need %d). Waiting.", len(bars), TDA_MIN_BARS)
            ib.disconnect()
            save_state(state)
            return

        # Compute signal
        signal = compute_intraday_signal(bars)
        regime = signal["regime"]
        action = signal["action"]
        current_price = signal["current_price"]
        atr = signal["atr"]

        log.info("Signal: %s (regime=%s, %s)", action, regime_name(regime), signal["reason"])

        # Check flatten time (3:55 PM ET)
        # Server runs in UTC. Convert to EDT (UTC-4) for flatten time check.
        from datetime import timezone as tz_cls
        edt_offset = tz_cls(timedelta(hours=-4))
        now_edt = datetime.now(edt_offset)
        if now_edt.time() >= FLATTEN_TIME and state["position"] > 0:
            log.info("Flatten time reached (3:55 PM). Closing intraday position.")
            fill = await close_intraday(ib, contract, state["position"], "EOD flatten")
            if fill > 0:
                pnl = (fill - state["entry_price"]) * state["position"] * 5  # MES $5/pt
                state["intraday_pnl"] += pnl
                state["intraday_trades"] += 1
                state["position"] = 0
                discord(f"EOD FLATTEN: closed {state['position']} MES @ {fill:.2f} P&L=${pnl:.2f}")
            ib.disconnect()
            save_state(state)
            return

        # Execute based on signal
        if action == "BUY" and state["position"] == 0:
            # Enter long
            log.info("INTRADAY ENTRY: TDA trending up on 5-min bars")
            if dry_run:
                log.info("[DRY] Would BUY 1 MES @ ~%.2f", current_price)
            else:
                success = await place_intraday_bracket(ib, contract,
                                                        MAX_INTRADAY_CONTRACTS,
                                                        current_price, atr)
                if success:
                    state["position"] = MAX_INTRADAY_CONTRACTS
                    state["entry_price"] = current_price
                    state["intraday_trades"] += 1
                    discord(f"INTRADAY BUY 1 MES @ ~{current_price:.2f} | "
                            f"stop={tick_round(current_price - INTRADAY_ATR_STOP_MULT*atr):.2f} | "
                            f"sg={signal['sg']:.4f} mom={signal['mom_5bar']:.5f}")

        elif action == "EXIT" and state["position"] > 0:
            # Exit — regime shifted
            log.info("INTRADAY EXIT: regime changed to %s", regime_name(regime))
            if dry_run:
                log.info("[DRY] Would SELL %d MES @ ~%.2f", state["position"], current_price)
            else:
                fill = await close_intraday(ib, contract, state["position"],
                                             f"regime→{regime_name(regime)}")
                if fill > 0:
                    pnl = (fill - state["entry_price"]) * state["position"] * 5
                    state["intraday_pnl"] += pnl
                    state["position"] = 0
                    discord(f"INTRADAY EXIT {state['position']} MES @ {fill:.2f} P&L=${pnl:.2f} "
                            f"(regime={regime_name(regime)})")

        elif state["position"] > 0:
            # Holding — report status
            unrealized = (current_price - state["entry_price"]) * state["position"] * 5
            log.info("Holding %d MES | entry=%.2f | current=%.2f | unrealized=$%.2f",
                     state["position"], state["entry_price"], current_price, unrealized)

        else:
            log.info("Flat. Waiting for trending signal. (sg=%.4f, p25=%.4f)",
                     signal["sg"], signal.get("sg_p25", 0))

        # Track regime history for analysis
        state["regime_history"].append({
            "time": now.strftime("%H:%M"),
            "regime": regime_name(regime),
            "sg": round(signal["sg"], 4),
            "action": action,
        })
        state["regime_history"] = state["regime_history"][-20:]

        ib.disconnect()

    except Exception as e:
        log.error("Scan error: %s", e, exc_info=True)
        try:
            ib.disconnect()
        except Exception:
            pass

    save_state(state)
    log.info("Scan complete. Trades today: %d | P&L: $%.2f",
             state["intraday_trades"], state["intraday_pnl"])


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("DRY RUN")
    asyncio.run(run_scan(dry_run=dry))
