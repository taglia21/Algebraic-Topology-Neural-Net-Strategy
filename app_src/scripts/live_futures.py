#!/usr/bin/env python3
"""
live_futures.py — TDA + TCN on MES Micro Futures (Fixed)
=========================================================
Fixes applied vs prior version:
  1. regime_to_contracts() now actually drives position sizing (not dead code)
  2. Normalization uses saved training mean/std (not inference window stats)
  3. Wasserstein state persisted to disk (not reset every invocation)
  4. Atomic state file writes (no corruption on kill signal)
  5. IBS overlay tracked correctly with its own state flag
  6. Confidence gating: min 0.55 to act (above 72% majority-class floor)

Trading vehicle: MES (Micro E-Mini S&P 500)
  - $5 per SPX point
  - No PDT rule (CFTC regulated, not FINRA)
  - Commission: ~$0.50 round-trip per contract
  - Overnight margin: ~$1,200-1,500 per contract

Run: PYTHONPATH=/opt/atnn/app_src python3 /opt/atnn/scripts/live_futures.py
"""

from __future__ import annotations
import asyncio, json, logging, os, sys, tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

APP_SRC = Path(__file__).resolve().parent.parent / "app_src"
if not (APP_SRC / "tda").exists():
    APP_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_SRC))

import yfinance as yf
import torch

from tda.extractor import TDAFeatureExtractor
from nn.models.tcn_predictor import TCNPredictor
from nn.regime_labeler import regime_to_contracts, regime_name, label_regimes

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

MES_SYMBOL   = "MES"
MES_EXCHANGE = "CME"

MAX_CONTRACTS   = 2
MIN_CONFIDENCE  = 0.55    # must exceed majority-class floor (~0.50 for balanced)

MODEL_PATH    = Path("/opt/atnn/models/tcn_tda_model.pt")
STATE_FILE    = Path("/opt/atnn/data/futures_state.json")
WASS_DIAG_FILE = Path("/opt/atnn/data/last_h1_diagram.npy")  # persistence for wasserstein

FEAT_COLS = [
    "beta_0", "beta_1", "persistence_entropy", "wasserstein_dist",
    "spectral_gap", "sci",
    "mom_5", "mom_20", "vol_10", "rsi", "log_ret",
]

# IBS overlay parameters
IBS_ATR_WINDOW  = 25
IBS_HIGH_WINDOW = 10
IBS_ATR_MULT    = 2.5
IBS_THRESH      = 0.30


# ─── Atomic State I/O ─────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"day": 0, "ibs_active": False, "pnl_today": 0.0, "regime_history": []}


def save_state(s: dict):
    """Atomic write — avoids state corruption on kill signal."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(s, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)   # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# ─── Signals ──────────────────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_ibs(spy_df: pd.DataFrame) -> tuple[bool, bool, float]:
    """
    Returns (should_enter, should_exit, ibs_value).

    Entry: Close < (10-day high - 2.5 * 25-day avg range) AND IBS < 0.30
    Exit:  Close > previous day's high
    """
    h = spy_df["High"].squeeze()
    l = spy_df["Low"].squeeze()
    c = spy_df["Close"].squeeze()

    if len(c) < IBS_ATR_WINDOW + 5:
        return False, False, 0.5

    ibs_val   = float((c.iloc[-1] - l.iloc[-1]) / max(h.iloc[-1] - l.iloc[-1], 0.001))
    avg_rng   = float((h - l).rolling(IBS_ATR_WINDOW).mean().iloc[-1])
    roll_high = float(h.rolling(IBS_HIGH_WINDOW).max().iloc[-1])
    threshold = roll_high - IBS_ATR_MULT * avg_rng

    enter = (float(c.iloc[-1]) < threshold) and (ibs_val < IBS_THRESH)
    exit_ = len(c) >= 2 and float(c.iloc[-1]) > float(h.iloc[-2])

    return enter, exit_, ibs_val


def compute_tcn_regime(spy_df: pd.DataFrame) -> tuple[int, float]:
    """
    Run TDA feature extraction + TCN inference.

    Returns (regime_class, confidence) using:
    1. Saved training normalization (if model exists)
    2. Heuristic fallback (if no model)
    """
    close = spy_df["Close"].squeeze().dropna()
    high  = spy_df["High"].squeeze().dropna()

    if len(close) < 80:
        return 2, 0.4   # default: mean-reverting

    # ── TDA features ──
    # Load or initialize the persisted H1 diagram for wasserstein continuity
    extractor = TDAFeatureExtractor(window=40, stride=1)
    if WASS_DIAG_FILE.exists():
        try:
            extractor._prev_h1_diagram = np.load(str(WASS_DIAG_FILE), allow_pickle=True)
        except Exception:
            pass

    tda = extractor.extract_series(close)

    # Persist the H1 diagram for next run (fixes always-zero wasserstein bug)
    if extractor._prev_h1_diagram is not None:
        WASS_DIAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(WASS_DIAG_FILE), extractor._prev_h1_diagram, allow_pickle=True)

    if len(tda) < 5:
        return 2, 0.4

    # ── Price features ──
    log_ret = np.log(close / close.shift(1))
    price_feats = pd.DataFrame({
        "mom_5":   close.pct_change(5),
        "mom_20":  close.pct_change(20),
        "vol_10":  log_ret.rolling(10).std() * np.sqrt(252),
        "rsi":     _rsi(close, 14) / 100.0,
        "log_ret": log_ret,
    })

    # Align to TDA index
    available_cols = [c for c in FEAT_COLS if c in tda.columns or c in price_feats.columns]
    tda_cols   = [c for c in available_cols if c in tda.columns]
    price_cols = [c for c in available_cols if c in price_feats.columns]

    combined = pd.concat([tda[tda_cols], price_feats[price_cols]], axis=1).dropna()
    if len(combined) < 5:
        return 2, 0.4

    last_row = combined.iloc[-1]

    log.info("TDA (last bar): beta_0=%.1f beta_1=%.1f entropy=%.3f wass=%.3f spec_gap=%.3f",
             last_row.get("beta_0", 0), last_row.get("beta_1", 0),
             last_row.get("persistence_entropy", 0),
             last_row.get("wasserstein_dist", 0),
             last_row.get("spectral_gap", 0.5))

    # ── TCN prediction ──
    SEQ_LEN = 30
    if MODEL_PATH.exists() and len(combined) >= SEQ_LEN:
        try:
            ckpt  = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
            model = TCNPredictor(
                input_size=ckpt["n_features"],
                num_channels=[64, 64, 32],
                num_classes=4,
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            # Use TRAINING normalization stats (not inference window stats)
            feat_mean = ckpt.get("feat_mean")
            feat_std  = ckpt.get("feat_std")

            feat_names = ckpt.get("feature_names", available_cols)
            avail = [c for c in feat_names if c in combined.columns]
            vals = combined[avail].tail(SEQ_LEN).values.astype(np.float32)

            if feat_mean is not None and feat_std is not None:
                vals = (vals - feat_mean[:len(avail)]) / feat_std[:len(avail)]
            else:
                # Fallback: window normalization (less accurate)
                mean_v = vals.mean(0); std_v = vals.std(0)
                std_v[std_v == 0] = 1.0
                vals = (vals - mean_v) / std_v

            vals = np.nan_to_num(vals)
            x    = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(x), dim=-1)
            regime     = int(probs.argmax(1).item())
            confidence = float(probs.max(1).values.item())

            log.info("TCN: regime=%s confidence=%.3f", regime_name(regime), confidence)
            return regime, confidence

        except Exception as e:
            log.warning("TCN inference failed (%s), using heuristic", e)

    # ── Heuristic fallback (no trained model yet) ──
    return _heuristic_regime(tda, close)


def _heuristic_regime(tda: pd.DataFrame, close: pd.Series) -> tuple[int, float]:
    """Heuristic regime from TDA + price. Used before first model is trained."""
    last = tda.iloc[-1]
    beta_1   = float(last.get("beta_1", 0))
    wass     = float(last.get("wasserstein_dist", 0))
    spec_gap = float(last.get("spectral_gap", 0.5))

    mom_5 = float(close.pct_change(5).iloc[-1]) if len(close) >= 5 else 0
    log_ret = np.log(close / close.shift(1))
    vol = float(log_ret.rolling(10).std().iloc[-1]) if len(close) >= 10 else 0.02

    # High wasserstein → regime transition in progress
    if wass > 0.5:
        return 3, 0.60   # volatile / transition

    # Low spectral gap → correlated, trending market
    if spec_gap < 0.3:
        return (0, 0.58) if mom_5 > 0.01 else (1, 0.58) if mom_5 < -0.01 else (2, 0.50)

    # High beta_1 → many loops in topology = oscillating / mean-reverting
    if beta_1 > 3.0:
        return 2, 0.62

    # High vol
    if vol > 0.025:
        return 3, 0.60

    return 2, 0.48   # default


# ─── IBKR Execution ───────────────────────────────────────────────────────────

async def get_mes_price(ib) -> float:
    """Get current MES mid-price from IBKR market data."""
    try:
        from ib_async import Future, ContFuture
        contract = Future(MES_SYMBOL, exchange=MES_EXCHANGE, currency="USD")
        qs = await ib.reqMktDataAsync(contract, "", False, False)
        await asyncio.sleep(2)
        ticker = ib.ticker(contract)
        if ticker and ticker.midpoint():
            return float(ticker.midpoint())
    except Exception:
        pass
    # Fallback: use last SPY close × 10 as SPX proxy
    import yfinance as yf
    spy = yf.download("SPY", period="5d", interval="1d", auto_adjust=True, progress=False)
    return float(spy["Close"].squeeze().iloc[-1]) * 10


async def trade_mes(ib, action: str, qty: int, dry_run: bool = False) -> bool:
    """Place MES market order. Returns True if filled."""
    if qty <= 0:
        return True
    from ib_async import Future, MarketOrder

    contract = Future(MES_SYMBOL, exchange=MES_EXCHANGE, currency="USD")
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            log.error("Cannot qualify MES contract — futures permissions may not be enabled")
            log.error("Enable: IBKR Client Portal → Settings → Account Settings → Futures")
            return False
        contract = qualified[0]
    except Exception as e:
        log.error("MES qualification failed: %s", e)
        return False

    log.info("[%s] %s %d MES (%s)", "DRY" if dry_run else "LIVE", action, qty,
             contract.localSymbol)
    if dry_run:
        return True

    order = MarketOrder(action, qty)
    trade = ib.placeOrder(contract, order)

    for _ in range(120):
        await asyncio.sleep(0.5)
        if trade.isDone():
            status = trade.orderStatus.status
            filled = trade.orderStatus.filled
            price  = trade.orderStatus.avgFillPrice
            log.info("MES order %s: %d @ %.2f", status, int(filled), price)
            return status == "Filled"

    log.warning("MES order timeout")
    return False


# ─── Main Cycle ───────────────────────────────────────────────────────────────

async def run_cycle(dry_run: bool = False):
    state = load_state()
    state["day"] = state.get("day", 0) + 1
    log.info("=" * 60)
    log.info("Day %d | %s", state["day"], datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # 1. Fetch data
    log.info("Fetching SPY data...")
    spy_raw = yf.download("SPY", period="400d", interval="1d",
                          auto_adjust=True, progress=False)
    if spy_raw is None or len(spy_raw) < 80:
        log.error("SPY data unavailable. Aborting.")
        save_state(state)
        return

    # Check data freshness (yfinance can return stale data)
    last_bar = spy_raw.index[-1].date()
    today    = datetime.now().date()
    if (today - last_bar).days > 5:
        log.warning("SPY data may be stale (last bar: %s, today: %s)", last_bar, today)

    # 2. Compute signals
    log.info("Computing IBS signal...")
    ibs_enter, ibs_exit, ibs_val = compute_ibs(spy_raw)
    ibs_was_active = state.get("ibs_active", False)

    log.info("Computing TDA + TCN regime...")
    regime, confidence = compute_tcn_regime(spy_raw)
    log.info("Regime: %s (confidence=%.3f, min=%.2f)", regime_name(regime),
             confidence, MIN_CONFIDENCE)

    # 3. Connect to IBKR
    from ib_async import IB
    ib = IB()
    try:
        await asyncio.wait_for(
            ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID), timeout=20
        )
        log.info("Connected to IBKR")
    except Exception as e:
        log.error("IBKR connection failed: %s", e)
        save_state(state)
        return

    try:
        acct = await ib.accountSummaryAsync()
        nav  = float(next((s.value for s in acct if s.tag == "NetLiquidation"), 5923))
        log.info("NAV: $%.2f", nav)

        # Current MES positions
        current_mes = 0
        for pos in ib.positions():
            if pos.contract.symbol == MES_SYMBOL and pos.contract.secType in ("FUT", "CONTFUT"):
                current_mes = int(pos.position)
        log.info("Current MES position: %d", current_mes)

        # 4. Determine target

        # IBS overlay (highest priority — validated signal)
        if ibs_was_active and ibs_exit:
            log.info("IBS EXIT: closing IBS-driven position")
            state["ibs_active"] = False
            # Fall through to regime-driven target
            ibs_was_active = False

        if ibs_enter and not ibs_was_active:
            log.info("IBS ENTRY: SPY oversold (IBS=%.3f), entering long MES", ibs_val)
            target_qty = MAX_CONTRACTS
            state["ibs_active"] = True

        elif ibs_was_active:
            # Still in IBS trade, hold
            target_qty = MAX_CONTRACTS
            log.info("IBS: holding (IBS=%.3f)", ibs_val)

        else:
            # TCN regime drives position
            # regime_to_contracts() uses confidence gating and regime logic
            target_qty = regime_to_contracts(
                regime=regime,
                confidence=confidence,
                nav=nav,
                mes_price=spy_raw["Close"].squeeze().iloc[-1] * 10,
                max_contracts=MAX_CONTRACTS,
                min_confidence=MIN_CONFIDENCE,
            )

        log.info("Target MES: %d (current: %d)", target_qty, current_mes)

        # 5. Execute
        if target_qty > current_mes:
            delta = target_qty - current_mes
            success = await trade_mes(ib, "BUY", delta, dry_run=dry_run)
            if not success and not dry_run:
                log.error("BUY order failed")
        elif target_qty < current_mes:
            delta = current_mes - target_qty
            success = await trade_mes(ib, "SELL", delta, dry_run=dry_run)
            if not success and not dry_run:
                log.error("SELL order failed")
        else:
            log.info("No position change needed")

        # 6. Update state
        state["regime_history"].append({
            "date": str(today),
            "regime": regime_name(regime),
            "confidence": round(confidence, 3),
            "target_mes": target_qty,
            "ibs_val": round(ibs_val, 3),
        })
        state["regime_history"] = state["regime_history"][-60:]

        # 7. Discord notification
        try:
            import subprocess
            msg = (f"ATNN Day {state['day']}: "
                   f"Regime={regime_name(regime)} conf={confidence:.2f} | "
                   f"MES={target_qty}cts | "
                   f"IBS={'ACTIVE' if state.get('ibs_active') else 'flat'}({ibs_val:.3f}) | "
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
    log.info("Done. State saved atomically.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("DRY RUN — no real orders")
    asyncio.run(run_cycle(dry_run=dry))
