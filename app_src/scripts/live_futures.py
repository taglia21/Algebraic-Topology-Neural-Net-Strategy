#!/usr/bin/env python3
"""
live_futures.py — TDA + TCN strategy on MES Micro Futures
===========================================================
This is the correct trading vehicle:
  - MES (Micro E-Mini S&P 500): tracks S&P 500
  - MNQ (Micro E-Mini Nasdaq): tracks Nasdaq 100
  - NO PDT rule (CFTC regulated, not FINRA)
  - Unlimited day trades
  - Commission: $0.25-0.85/contract (vs $1 min for stocks)
  - Overnight margin: ~$1,200-1,500/contract (can hold 3-4 on $5,923)
  - Same IBKR API and gateway as stocks

Strategy:
  1. Use TDA (topological data analysis) to detect market regime
  2. Use TCN to predict regime from TDA + price features
  3. Execute:
     - TRENDING_UP regime  → long 2 MES contracts
     - TRENDING_DOWN regime → flat (or short 1 if model confident)
     - MEAN_REVERTING      → scalp mean-reversion entries
     - VOLATILE            → flat, wait

Prerequisites:
  - Enable futures trading in IBKR Client Portal:
    Settings → Account Settings → Trading Experiences & Permissions → Futures
  - Run this script daily at 9:35 AM ET after 2FA approval

Run: python3 /opt/atnn/scripts/live_futures.py
"""

from __future__ import annotations
import asyncio, json, logging, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

APP_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(APP_SRC))

import yfinance as yf
import torch

from tda.extractor import TDAFeatureExtractor
from nn.models.tcn_predictor import TCNPredictor
from nn.regime_labeler import regime_to_strategy_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_futures")

# ─── Config ───────────────────────────────────────────────────────────────────

IBKR_HOST  = "127.0.0.1"
IBKR_PORT  = 4003
CLIENT_ID  = 30
ACCOUNT    = "U22452226"

MES_SYMBOL    = "MES"
MNQ_SYMBOL    = "MNQ"
MES_EXCHANGE  = "CME"
MES_CURRENCY  = "USD"

MAX_CONTRACTS = 2          # Max MES contracts to hold
STOP_TICKS    = 20         # 20 ticks = $50 per MES contract (5 pts × $5)
TARGET_TICKS  = 40         # 40 ticks = $100 per MES contract
MES_TICK_SIZE = 0.25       # 0.25 SPX points per tick
MES_MULTIPLIER = 5.0       # $5 per SPX point

MODEL_PATH = Path("/opt/atnn/models/tcn_tda_model.pt")
STATE_FILE = Path("/opt/atnn/data/futures_state.json")

# ─── State ────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"day": 0, "regime_history": [], "pnl_today": 0.0}

def save_state(s: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))

# ─── Signals ──────────────────────────────────────────────────────────────────

def compute_signals(prices_df: pd.DataFrame) -> dict:
    """
    Compute regime signal from TDA + price features.
    Returns dict with regime, confidence, raw features, and IBS overlay.
    """
    close = prices_df["Close"].dropna()
    high  = prices_df["High"].dropna()
    low   = prices_df["Low"].dropna()

    if len(close) < 80:
        return {"regime": 3, "confidence": 0.5, "regime_name": "VOLATILE", "ibs_entry": False}

    # ── TDA features ──
    extractor = TDAFeatureExtractor(window=40, stride=1)
    tda_series = extractor.extract_series(close)

    # ── Price features ──
    log_ret   = np.log(close / close.shift(1)).dropna()
    mom_5     = close.pct_change(5).dropna()
    mom_20    = close.pct_change(20).dropna()
    vol_10    = log_ret.rolling(10).std().dropna() * np.sqrt(252)
    rsi_raw   = _rsi(close, 14)

    # ── Combine features (last N bars) ──
    SEQ_LEN = 30
    feat_cols = ["beta_0", "beta_1", "persistence_entropy", "wasserstein_dist",
                 "spectral_gap", "sci"]

    if len(tda_series) < SEQ_LEN:
        log.warning("Not enough TDA features (%d, need %d)", len(tda_series), SEQ_LEN)
        return {"regime": 2, "confidence": 0.4, "regime_name": "MEAN_REVERTING", "ibs_entry": False}

    # Align all features to TDA index
    tda_tail = tda_series[feat_cols].tail(SEQ_LEN)
    common   = tda_tail.index

    price_feats = pd.DataFrame({
        "mom_5":   mom_5.reindex(common).ffill().bfill(),
        "mom_20":  mom_20.reindex(common).ffill().bfill(),
        "vol_10":  vol_10.reindex(common).ffill().bfill(),
        "rsi":     rsi_raw.reindex(common).ffill().bfill() / 100.0,
        "log_ret": log_ret.reindex(common).ffill().bfill(),
    }, index=common)

    full_feats = pd.concat([tda_tail, price_feats], axis=1).dropna()

    if len(full_feats) < SEQ_LEN // 2:
        return {"regime": 2, "confidence": 0.4, "regime_name": "MEAN_REVERTING", "ibs_entry": False}

    n_features = full_feats.shape[1]

    # ── TCN prediction (if model exists) ──
    regime, confidence = 2, 0.5  # default: mean-reverting

    if MODEL_PATH.exists():
        try:
            ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
            model = TCNPredictor(input_size=ckpt.get("n_features", n_features))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            # Normalize
            vals = full_feats.values.astype(np.float32)
            mean_v = vals.mean(axis=0)
            std_v  = vals.std(axis=0)
            std_v[std_v == 0] = 1.0
            vals = (vals - mean_v) / std_v
            vals = np.nan_to_num(vals)

            x = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)  # (1, seq, feats)
            cls, conf = model.predict_regime(x)
            regime     = int(cls[0].item())
            confidence = float(conf[0].item())
        except Exception as e:
            log.warning("TCN prediction failed (%s), using heuristic regime", e)

    # If no trained model, use heuristic regime from TDA + price signals
    if not MODEL_PATH.exists():
        regime, confidence = _heuristic_regime(tda_series.tail(5), close, vol_10)

    # ── IBS mean-reversion overlay ──
    # Enter long if SPY is extremely oversold, regardless of regime
    last_ibs = float((close.iloc[-1] - low.iloc[-1]) /
                     max(high.iloc[-1] - low.iloc[-1], 0.001))
    avg_rng   = (high - low).rolling(25).mean().iloc[-1]
    roll_high = high.rolling(10).max().iloc[-1]
    ibs_thr   = float(roll_high - 2.5 * avg_rng)
    ibs_entry = (close.iloc[-1] < ibs_thr) and (last_ibs < 0.30)

    regime_names = {0: "TRENDING_UP", 1: "TRENDING_DOWN",
                    2: "MEAN_REVERTING", 3: "VOLATILE"}

    log.info("TDA features (last bar):")
    if len(tda_series) > 0:
        last = tda_series.iloc[-1]
        log.info("  beta_0=%.2f beta_1=%.2f entropy=%.3f wass=%.3f spec_gap=%.3f",
                 last.get("beta_0", 0), last.get("beta_1", 0),
                 last.get("persistence_entropy", 0),
                 last.get("wasserstein_dist", 0),
                 last.get("spectral_gap", 0))

    return {
        "regime": regime,
        "confidence": confidence,
        "regime_name": regime_names.get(regime, "UNKNOWN"),
        "ibs_entry": ibs_entry,
        "ibs_value": last_ibs,
        "n_features": n_features,
    }


def _heuristic_regime(tda_tail: pd.DataFrame, close: pd.Series,
                       vol_10: pd.Series) -> tuple[int, float]:
    """
    Heuristic regime detection from TDA features (no trained model needed).
    Used on first run before model is trained.
    """
    if len(tda_tail) == 0:
        return 2, 0.4

    last = tda_tail.iloc[-1]
    beta_1   = float(last.get("beta_1", 0))
    wass     = float(last.get("wasserstein_dist", 0))
    spec_gap = float(last.get("spectral_gap", 0.5))

    mom_5 = float(close.pct_change(5).iloc[-1]) if len(close) >= 5 else 0
    rv    = float(vol_10.iloc[-1]) if len(vol_10) > 0 else 0.15

    # Regime switch detected (high wasserstein distance)
    if wass > 1.0:
        return 3, 0.65  # volatile/transition

    # Trending: low spectral gap (high correlation), clear momentum
    if spec_gap < 0.3 and mom_5 > 0.01:
        return 0, 0.60  # trending up
    if spec_gap < 0.3 and mom_5 < -0.01:
        return 1, 0.60  # trending down

    # Mean-reverting: high beta_1 (loops in topology)
    if beta_1 > 2.0:
        return 2, 0.65

    # High vol
    if rv > 0.25:
        return 3, 0.60

    return 2, 0.45  # default: mean-reverting


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ─── IBKR Execution ───────────────────────────────────────────────────────────

async def connect_ibkr():
    from ib_async import IB
    ib = IB()
    await asyncio.wait_for(
        ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID), timeout=20
    )
    log.info("Connected to IBKR (server v%s)", ib.serverVersion())
    return ib


async def get_futures_positions(ib) -> dict:
    """Returns {symbol: net_qty} for futures positions."""
    return {
        p.contract.symbol: int(p.position)
        for p in ib.positions()
        if p.contract.secType in ("FUT", "CONTFUT") and int(p.position) != 0
    }


async def trade_futures(
    ib, symbol: str, action: str, qty: int, dry_run: bool = False
) -> bool:
    """Place a market order for futures contract."""
    if qty <= 0:
        return True

    from ib_async import Future, MarketOrder

    # Build front-month continuous contract
    contract = Future(symbol, exchange=MES_EXCHANGE, currency=MES_CURRENCY)

    # Qualify the contract to get the actual expiry
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            log.error("Could not qualify contract %s", symbol)
            return False
        contract = qualified[0]
        log.info("Contract: %s %s %s", contract.symbol, contract.lastTradeDateOrContractMonth,
                 contract.localSymbol)
    except Exception as e:
        log.error("Contract qualification failed: %s", e)
        return False

    order = MarketOrder(action, qty)
    log.info("[%s] %s %d %s", "DRY" if dry_run else "LIVE", action, qty, symbol)

    if dry_run:
        return True

    trade = ib.placeOrder(contract, order)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if trade.isDone():
            filled = trade.orderStatus.filled
            price  = trade.orderStatus.avgFillPrice
            status = trade.orderStatus.status
            log.info("Order %s: %d filled @ %.2f", status, int(filled), price)
            return status == "Filled"

    log.warning("Order timeout for %s", symbol)
    return False


# ─── Main cycle ───────────────────────────────────────────────────────────────

async def run_cycle(dry_run: bool = False):
    state = load_state()
    state["day"] = state.get("day", 0) + 1
    log.info("=" * 60)
    log.info("Day %d | %s | DRY=%s", state["day"],
             datetime.now().strftime("%Y-%m-%d %H:%M"), dry_run)
    log.info("=" * 60)

    # 1. Get price data (use SPY as proxy for MES signal)
    log.info("Fetching price data...")
    spy_raw = yf.download("SPY", period="400d", interval="1d",
                          auto_adjust=True, progress=False)
    if spy_raw is None or len(spy_raw) < 80:
        log.error("Insufficient price data. Aborting.")
        save_state(state)
        return

    # 2. Compute signals
    log.info("Computing TDA + TCN regime signal...")
    signal = compute_signals(spy_raw)
    regime     = signal["regime"]
    confidence = signal["confidence"]
    regime_nm  = signal["regime_name"]
    ibs_entry  = signal["ibs_entry"]

    log.info("Regime: %s (conf=%.2f)", regime_nm, confidence)
    log.info("IBS entry: %s (IBS=%.3f)", ibs_entry, signal.get("ibs_value", 0))

    weights = regime_to_strategy_weights(regime)
    log.info("Strategy weights: %s", weights)

    # 3. Connect IBKR
    try:
        ib = await connect_ibkr()
    except Exception as e:
        log.error("IBKR connect failed: %s", e)
        save_state(state)
        return

    try:
        acct = await ib.accountSummaryAsync()
        nav  = float(next((s.value for s in acct if s.tag == "NetLiquidation"), 5923))
        positions = await get_futures_positions(ib)
        log.info("NAV: $%.2f | Futures positions: %s", nav, positions)

        current_mes = positions.get(MES_SYMBOL, 0)

        # 4. Determine target position
        if ibs_entry or regime == 0:
            # Trending up or IBS oversold → long
            target_qty = MAX_CONTRACTS
        elif regime == 1:
            # Trending down → flat (can short later when model is trained + confident)
            target_qty = 0
        elif regime == 2:
            # Mean-reverting → small long (MES range trading)
            target_qty = 1 if confidence > 0.60 else 0
        else:
            # Volatile → flat
            target_qty = 0

        # Scale by confidence
        if not ibs_entry:
            target_qty = max(0, round(target_qty * min(confidence / 0.65, 1.0)))

        log.info("Current MES: %d | Target MES: %d", current_mes, target_qty)

        # 5. Execute
        if target_qty > current_mes:
            delta = target_qty - current_mes
            await trade_futures(ib, MES_SYMBOL, "BUY", delta, dry_run=dry_run)
        elif target_qty < current_mes:
            delta = current_mes - target_qty
            action = "SELL" if current_mes > 0 else "BUY"  # cover short
            await trade_futures(ib, MES_SYMBOL, action, abs(delta), dry_run=dry_run)
        else:
            log.info("No change needed.")

        # 6. Update state
        state["regime_history"].append({
            "date": str(datetime.now().date()),
            "regime": regime_nm,
            "confidence": round(confidence, 3),
            "target_qty": target_qty,
        })
        state["regime_history"] = state["regime_history"][-30:]  # keep last 30

        # 7. Discord notification
        try:
            import subprocess
            msg = (f"ATNN Day {state['day']}: "
                   f"Regime={regime_nm} conf={confidence:.2f} | "
                   f"MES target={target_qty} | "
                   f"IBS={'ENTRY' if ibs_entry else 'flat'} | "
                   f"NAV=${nav:.2f}")
            subprocess.run([
                "curl", "-s", "-X", "POST",
                "-H", "Content-Type: application/json",
                "-H", "User-Agent: curl/7.68.0",
                "-d", json.dumps({"content": msg}),
                "https://discord.com/api/webhooks/1482171912724545638/"
                "EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1",
            ], capture_output=True, timeout=10)
        except Exception:
            pass

        ib.disconnect()

    except Exception as e:
        log.error("Cycle error: %s", e, exc_info=True)
        try:
            ib.disconnect()
        except Exception:
            pass

    save_state(state)
    log.info("Cycle complete.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("DRY RUN — no real orders placed")
    asyncio.run(run_cycle(dry_run=dry))
