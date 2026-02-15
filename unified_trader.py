#!/usr/bin/env python3
"""
Unified Production Trader — Single Entry Point
================================================

This is the ONE production trading bot. It replaces profit_trader.py,
smart_trader.py, continuous_trader.py, etc.

Wires together ALL sophisticated modules from src/:
  - Regime Detection   (HMM + GMM ensemble)
  - TDA Engine         (Persistent homology confirmation)
  - Signal Aggregation (Multi-model Bayesian ensemble)
  - ML Predictions     (LSTM neural net confidence)
  - ATR Stop Losses    (Dynamic volatility-based stops)
  - Half-Kelly Sizing  (Optimal position sizing, 5% cap)
  - Sector Caps        (Max 25% per sector, 2 positions)
  - Circuit Breaker    (3% daily loss halt)
  - Process Lock       (Single instance enforcement)

Data: Alpaca REST API ONLY. Zero yfinance.
Orders: Limit orders ONLY. Zero market orders.

Usage:
    python unified_trader.py
    python unified_trader.py --dry-run
    python unified_trader.py --scan-only
"""

import os
import sys
import json
import time
import signal
import atexit
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, IO
from enum import Enum
import threading
import pickle
from collections import deque

import numpy as np
import requests
from dotenv import load_dotenv

# ============================================================================
# LOAD ENVIRONMENT
# ============================================================================
load_dotenv()

# ============================================================================
# LOGGING — console + file
# ============================================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("unified_trader")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Console handler (INFO+)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
))
logger.addHandler(_ch)

# File handler (DEBUG+)
_fh = logging.FileHandler(LOG_DIR / "unified_trader.log", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
))
logger.addHandler(_fh)

# ============================================================================
# ALPACA REST CLIENT — pure requests, no SDK
# ============================================================================
ALPACA_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA = "https://data.alpaca.markets"

if not ALPACA_KEY or not ALPACA_SECRET:
    logger.error("Missing Alpaca API credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
    sys.exit(1)

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "Content-Type": "application/json",
}

# ============================================================================
# TRADING UNIVERSE — 35 symbols across 8 sectors
# ============================================================================
UNIVERSE = [
    # Technology (6)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "CRM",
    # Healthcare (4)
    "UNH", "JNJ", "LLY", "ABBV",
    # Financials (4)
    "JPM", "GS", "V", "MA",
    # Energy (3)
    "XOM", "CVX", "COP",
    # Consumer Discretionary (3)
    "AMZN", "TSLA", "NFLX",
    # Consumer Staples (3)
    "KO", "PG", "COST",
    # Industrials (4)
    "CAT", "HON", "GE", "DE",
    # REITs (2)
    "AMT", "O",
    # ETFs (6)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLV",
]

# ============================================================================
# SECTOR MAP — enforced, NOT optional
# ============================================================================
SECTOR_MAP = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "GOOGL": "technology", "META": "technology", "CRM": "technology",
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare", "ABBV": "healthcare",
    "JPM": "financials", "GS": "financials", "V": "financials", "MA": "financials",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "AMZN": "consumer", "TSLA": "consumer", "NFLX": "consumer",
    "KO": "staples", "PG": "staples", "COST": "staples",
    "CAT": "industrials", "HON": "industrials", "GE": "industrials", "DE": "industrials",
    "AMT": "reits", "O": "reits",
    "SPY": "etf", "QQQ": "etf", "IWM": "etf",
    "XLF": "etf", "XLE": "etf", "XLV": "etf",
}
SECTOR_MAX_PCT = 0.25        # Max 25% equity per sector
SECTOR_MAX_POSITIONS = 2     # Max 2 positions per sector

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class UnifiedConfig:
    """All tunable parameters in one place."""
    # Scan timing
    scan_interval_sec: int = 300       # 5 min between scans
    market_open_hour: int = 9
    market_open_min: int = 35          # Wait 5 min after open
    market_close_hour: int = 15
    market_close_min: int = 50         # Stop 10 min before close

    # Position sizing (Half-Kelly, capped)
    max_position_pct: float = 0.05     # 5% max per position
    min_position_pct: float = 0.01     # 1% min
    kelly_fraction: float = 0.50       # Half-Kelly
    default_position_pct: float = 0.03 # 3% default before Kelly history

    # ATR stops
    atr_period: int = 14
    atr_mult_volatile: float = 2.0     # 2x ATR for volatile stocks
    atr_mult_stable: float = 1.5       # 1.5x ATR for stable stocks
    volatility_threshold: float = 0.02 # ATR% > 2% = volatile

    # Signal thresholds
    min_composite_score: float = 0.55  # Minimum score to buy
    min_tda_alignment: float = -0.2    # TDA must not be strongly negative

    # Profit taking
    profit_target_pct: float = 0.06    # 6% take profit
    trailing_stop_activation: float = 0.03  # Trail after 3% gain
    trailing_stop_pct: float = 0.015   # 1.5% trail distance

    # Circuit breaker
    max_daily_loss_pct: float = 0.03   # 3% daily loss halt
    max_open_positions: int = 12

    # Limit order buffer
    limit_buffer_pct: float = 0.001    # 0.1% above last for buys

    # Regime
    regime_cache_minutes: int = 15

    # Bars lookback
    bars_lookback: int = 100           # Days of bars to fetch

    # ── Options Integration ──────────────────────────────────────────
    options_enabled: bool = True
    options_underlyings: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    options_max_portfolio_pct: float = 0.20   # Max 20% of equity in options
    options_max_per_position_pct: float = 0.05  # Max 5% per options position
    options_target_dte: int = 45               # Target 45 DTE for premium selling
    options_min_dte_close: int = 7             # Close at 7 DTE
    options_take_profit_pct: float = 0.50      # Close at 50% of max profit
    options_stop_loss_mult: float = 2.0        # Stop at 2x credit received
    options_iv_rank_threshold: float = 50.0    # Only sell when IV rank > 50%
    options_scan_interval_min: int = 30        # Scan options every 30 min
    options_regimes_allowed: List[str] = field(
        default_factory=lambda: ["mean_reverting", "neutral", "trending_bull"]
    )

    # ── ML Hard Filter ───────────────────────────────────────────────
    ml_hard_filter: bool = True
    ml_min_confidence: float = 0.40            # Skip trade if ML conf < 0.4

    # ── Retraining ───────────────────────────────────────────────────
    retraining_enabled: bool = True
    retraining_hour_est: int = 0               # Midnight EST
    retraining_min_trades: int = 20            # Min trades before retraining

    # ── Thompson Sampling ────────────────────────────────────────────
    thompson_enabled: bool = True
    thompson_prior_alpha: float = 1.0          # Beta prior alpha (successes)
    thompson_prior_beta: float = 1.0           # Beta prior beta (failures)


# ============================================================================
# ALPACA API HELPERS
# ============================================================================

def alpaca_get(path: str, base: str = None, params: dict = None) -> Optional[dict]:
    """GET request to Alpaca API with error handling."""
    url = f"{base or ALPACA_BASE}{path}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            logger.debug(f"Alpaca 404: {path}")
            return None
        else:
            logger.warning(f"Alpaca {r.status_code}: {path} — {r.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Alpaca GET error {path}: {e}")
        return None


def alpaca_post(path: str, data: dict) -> Optional[dict]:
    """POST request to Alpaca API."""
    url = f"{ALPACA_BASE}{path}"
    try:
        r = requests.post(url, headers=HEADERS, json=data, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        else:
            logger.error(f"Alpaca POST {r.status_code}: {path} — {r.text[:300]}")
            return None
    except Exception as e:
        logger.error(f"Alpaca POST error {path}: {e}")
        return None


def alpaca_delete(path: str) -> bool:
    """DELETE request to Alpaca API."""
    url = f"{ALPACA_BASE}{path}"
    try:
        r = requests.delete(url, headers=HEADERS, timeout=10)
        return r.status_code in (200, 204)
    except Exception as e:
        logger.error(f"Alpaca DELETE error {path}: {e}")
        return False


def get_account() -> Optional[dict]:
    """Get Alpaca account info."""
    return alpaca_get("/v2/account")


def get_positions() -> List[dict]:
    """Get all open positions."""
    result = alpaca_get("/v2/positions")
    return result if isinstance(result, list) else []


def get_bars(symbol: str, timeframe: str = "1Day", limit: int = 100) -> Optional[List[dict]]:
    """Get historical bars from Alpaca data API."""
    params = {"timeframe": timeframe, "limit": limit, "adjustment": "split"}
    result = alpaca_get(f"/v2/stocks/{symbol}/bars", base=ALPACA_DATA, params=params)
    if result and "bars" in result:
        return result["bars"]
    return None


def get_latest_trade(symbol: str) -> Optional[float]:
    """Get latest trade price from Alpaca."""
    result = alpaca_get(f"/v2/stocks/{symbol}/trades/latest", base=ALPACA_DATA)
    if result and "trade" in result:
        return float(result["trade"]["p"])
    return None


def get_snapshot(symbol: str) -> Optional[dict]:
    """Get market snapshot for a symbol."""
    return alpaca_get(f"/v2/stocks/{symbol}/snapshot", base=ALPACA_DATA)


def submit_limit_order(symbol: str, qty: int, side: str, limit_price: float,
                       time_in_force: str = "day") -> Optional[dict]:
    """Submit a limit order. NEVER a market order."""
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "limit",
        "limit_price": str(round(limit_price, 2)),
        "time_in_force": time_in_force,
    }
    logger.info(f"ORDER: {side.upper()} {qty} {symbol} @ ${limit_price:.2f} LIMIT")
    return alpaca_post("/v2/orders", order)


def is_market_open() -> bool:
    """Check if market is currently open."""
    clock = alpaca_get("/v2/clock")
    if clock:
        return clock.get("is_open", False)
    return False


# ============================================================================
# OHLCV DATA FROM ALPACA BARS
# ============================================================================

def bars_to_arrays(bars: List[dict]) -> Dict[str, np.ndarray]:
    """Convert Alpaca bars to numpy arrays."""
    opens = np.array([float(b["o"]) for b in bars])
    highs = np.array([float(b["h"]) for b in bars])
    lows = np.array([float(b["l"]) for b in bars])
    closes = np.array([float(b["c"]) for b in bars])
    volumes = np.array([float(b["v"]) for b in bars])
    return {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}


# ============================================================================
# TECHNICAL INDICATORS — computed from Alpaca bar data  
# ============================================================================

def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI from close prices."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) + 1e-10
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_sma(closes: np.ndarray, period: int) -> float:
    """Simple moving average."""
    if len(closes) < period:
        return float(np.mean(closes))
    return float(np.mean(closes[-period:]))


def compute_ema(closes: np.ndarray, period: int) -> float:
    """Exponential moving average (last value)."""
    if len(closes) < period:
        return float(np.mean(closes))
    alpha = 2.0 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = alpha * c + (1 - alpha) * ema
    return float(ema)


def compute_macd(closes: np.ndarray) -> Tuple[float, float, float]:
    """MACD line, signal, histogram."""
    if len(closes) < 35:
        return 0.0, 0.0, 0.0
    ema12 = compute_ema(closes, 12)
    ema26 = compute_ema(closes, 26)
    macd_line = ema12 - ema26
    # Signal line: 9-period EMA of MACD
    # Approximate by computing full EMA series
    alpha12 = 2.0 / 13
    alpha26 = 2.0 / 27
    ema12_arr = np.zeros(len(closes))
    ema26_arr = np.zeros(len(closes))
    ema12_arr[0] = closes[0]
    ema26_arr[0] = closes[0]
    for i in range(1, len(closes)):
        ema12_arr[i] = alpha12 * closes[i] + (1 - alpha12) * ema12_arr[i-1]
        ema26_arr[i] = alpha26 * closes[i] + (1 - alpha26) * ema26_arr[i-1]
    macd_arr = ema12_arr - ema26_arr
    alpha9 = 2.0 / 10
    sig_arr = np.zeros(len(macd_arr))
    sig_arr[0] = macd_arr[0]
    for i in range(1, len(macd_arr)):
        sig_arr[i] = alpha9 * macd_arr[i] + (1 - alpha9) * sig_arr[i-1]
    histogram = macd_arr[-1] - sig_arr[-1]
    return float(macd_arr[-1]), float(sig_arr[-1]), float(histogram)


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> float:
    """Average True Range."""
    if len(closes) < period + 1:
        return float(np.mean(highs - lows)) if len(highs) > 0 else 0.01
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    return float(np.mean(tr[-period:]))


def compute_momentum(closes: np.ndarray, period: int = 10) -> float:
    """Momentum as % change over period."""
    if len(closes) < period + 1:
        return 0.0
    return float((closes[-1] / closes[-period - 1] - 1) * 100)


def compute_bollinger_position(closes: np.ndarray, period: int = 20) -> float:
    """Position within Bollinger Bands (0–1)."""
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mid = np.mean(window)
    std = np.std(window) + 1e-10
    upper = mid + 2 * std
    lower = mid - 2 * std
    return float(np.clip((closes[-1] - lower) / (upper - lower), 0, 1))


def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> float:
    """Average Directional Index."""
    if len(closes) < period + 2:
        return 20.0
    n = len(closes)
    plus_dm = np.zeros(n - 1)
    minus_dm = np.zeros(n - 1)
    tr = np.zeros(n - 1)
    for i in range(n - 1):
        high_diff = highs[i + 1] - highs[i]
        low_diff = lows[i] - lows[i + 1]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(highs[i+1] - lows[i+1],
                     abs(highs[i+1] - closes[i]),
                     abs(lows[i+1] - closes[i]))
    if len(tr) < period:
        return 20.0
    atr = np.mean(tr[:period])
    pdm_smooth = np.mean(plus_dm[:period])
    mdm_smooth = np.mean(minus_dm[:period])
    dx_list = []
    for i in range(period, len(tr)):
        atr = atr - (atr / period) + tr[i]
        pdm_smooth = pdm_smooth - (pdm_smooth / period) + plus_dm[i]
        mdm_smooth = mdm_smooth - (mdm_smooth / period) + minus_dm[i]
        if atr > 0:
            pdi = 100 * pdm_smooth / atr
            mdi = 100 * mdm_smooth / atr
        else:
            pdi = mdi = 0
        di_sum = pdi + mdi
        dx = 100 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0
        dx_list.append(dx)
    if not dx_list:
        return 20.0
    return float(min(np.mean(dx_list[-period:]), 100.0))


# ============================================================================
# TECHNICAL SIGNAL SCORING (smart_trader.py logic, expanded)
# ============================================================================

@dataclass
class TechnicalScore:
    """Technical analysis result for a symbol."""
    symbol: str
    price: float
    rsi: float
    momentum: float
    sma_5: float
    sma_15: float
    sma_50: float
    sma_200: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    atr: float
    atr_pct: float
    adx: float
    bollinger_pos: float
    volume_ratio: float
    score: float           # Composite technical score [0, 1]
    direction: str         # BUY, SELL, HOLD


def score_technicals(symbol: str, bars: List[dict], cfg: UnifiedConfig) -> Optional[TechnicalScore]:
    """Compute technical score from Alpaca bars, returns None on insufficient data."""
    if not bars or len(bars) < 50:
        return None

    d = bars_to_arrays(bars)
    closes = d["close"]
    highs = d["high"]
    lows = d["low"]
    volumes = d["volume"]
    price = float(closes[-1])

    rsi = compute_rsi(closes)
    mom = compute_momentum(closes, 10)
    sma5 = compute_sma(closes, 5)
    sma15 = compute_sma(closes, 15)
    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, min(200, len(closes)))
    macd_l, macd_s, macd_h = compute_macd(closes)
    atr = compute_atr(highs, lows, closes, cfg.atr_period)
    atr_pct = atr / price if price > 0 else 0.02
    adx = compute_adx(highs, lows, closes)
    bb_pos = compute_bollinger_position(closes)

    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    vol_ratio = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0

    # ── Scoring (0–1, higher = more bullish) ────────────────────────
    points = 0.0
    max_points = 10.0

    # RSI oversold = bullish
    if rsi < 30:
        points += 2.0
    elif rsi < 40:
        points += 1.0
    elif rsi > 70:
        points -= 1.0

    # Momentum positive
    if mom > 2.0:
        points += 1.5
    elif mom > 0.5:
        points += 0.5
    elif mom < -2.0:
        points -= 1.0

    # SMA crossover (5 > 15 = bullish)
    if sma5 > sma15:
        points += 1.0

    # Price above SMA50
    if price > sma50:
        points += 1.0

    # Price above SMA200
    if price > sma200:
        points += 0.5

    # MACD bullish
    if macd_h > 0:
        points += 1.0
    if macd_l > macd_s:
        points += 0.5

    # Bollinger: near bottom = bullish
    if bb_pos < 0.2:
        points += 1.0
    elif bb_pos > 0.9:
        points -= 0.5

    # Volume surge = confirmation
    if vol_ratio > 1.5:
        points += 0.5

    # ADX trending
    if adx > 25:
        points += 0.5

    # Normalize to [0, 1]
    score = float(np.clip((points + 3) / (max_points + 3), 0, 1))

    if score >= 0.6:
        direction = "BUY"
    elif score <= 0.35:
        direction = "SELL"
    else:
        direction = "HOLD"

    return TechnicalScore(
        symbol=symbol, price=price, rsi=rsi, momentum=mom,
        sma_5=sma5, sma_15=sma15, sma_50=sma50, sma_200=sma200,
        macd_line=macd_l, macd_signal=macd_s, macd_histogram=macd_h,
        atr=atr, atr_pct=atr_pct, adx=adx, bollinger_pos=bb_pos,
        volume_ratio=vol_ratio, score=score, direction=direction,
    )


# ============================================================================
# MODULE IMPORTS — graceful degradation with clear logging
# ============================================================================

# ── Regime Detector ─────────────────────────────────────────────────
_regime_detector = None
_RegimeEnum = None
try:
    from src.regime_detector import RuleBasedRegimeDetector, Regime as _RegimeEnumCls
    _regime_detector = RuleBasedRegimeDetector()
    _RegimeEnum = _RegimeEnumCls
    logger.info("✅ Regime detector loaded (HMM+GMM ensemble)")
except Exception as e:
    logger.warning(f"⚠️ Regime detector import failed: {e} — will use Alpaca-based fallback")

# ── TDA Engine ──────────────────────────────────────────────────────
_tda_engine = None
try:
    from src.tda_engine import TDAEngine
    _tda_engine = TDAEngine()
    logger.info("✅ TDA engine loaded (persistent homology)")
except Exception as e:
    logger.warning(f"⚠️ TDA engine import failed: {e} — TDA confirmation disabled")

# ── Signal Aggregator ───────────────────────────────────────────────
_signal_aggregator = None
try:
    from src.signal_aggregator import SignalAggregator, ModelSignal
    _signal_aggregator = SignalAggregator(min_confidence=0.4, min_models=1)
    logger.info("✅ Signal aggregator loaded (multi-model ensemble)")
except Exception as e:
    logger.warning(f"⚠️ Signal aggregator import failed: {e} — using technical-only signals")

# ── ML / Neural Net ─────────────────────────────────────────────────
_nn_predictor = None
try:
    from src.nn_predictor import NeuralNetPredictor
    _nn_predictor = NeuralNetPredictor()
    logger.info("✅ NN predictor loaded (LSTM)")
except Exception as e:
    logger.warning(f"⚠️ NN predictor import failed: {e} — ML signals disabled")

# ── ATR Stop Loss Manager ──────────────────────────────────────────
_stop_manager = None
try:
    from src.atr_stop_loss import DynamicStopLossManager, ATRCalculator, StopLossConfig
    _stop_config = StopLossConfig(
        atr_multiplier=2.0,
        atr_period=14,
        trailing=True,
        trail_activation_pct=0.03,
        max_loss_pct=0.10,
    )
    _stop_manager = DynamicStopLossManager(_stop_config)
    logger.info("✅ ATR stop-loss manager loaded")
except Exception as e:
    logger.warning(f"⚠️ ATR stop-loss import failed: {e} — using inline ATR stops")

# ── Kelly Position Sizer ───────────────────────────────────────────
_kelly_sizer = None
try:
    from src.kelly_position_sizer import KellyPositionSizer
    _kelly_sizer = KellyPositionSizer(
        min_position_pct=0.01,
        max_position_pct=0.05,
        kelly_fraction=0.50,
        volatility_scaling=True,
        target_volatility=0.15,
        min_trades_for_kelly=10,
        default_position_pct=0.03,
    )
    logger.info("✅ Kelly position sizer loaded (half-Kelly, 5% cap)")
except Exception as e:
    logger.warning(f"⚠️ Kelly sizer import failed: {e} — using fixed 3% sizing")

# ── Sector Caps ─────────────────────────────────────────────────────
# NON-OPTIONAL: we always enforce sector caps, using either src/ module or inline
_sector_module_available = False
try:
    from src.risk.sector_caps import sector_allows_trade as _src_sector_allows, get_sector as _src_get_sector
    _sector_module_available = True
    logger.info("✅ Sector caps module loaded")
except Exception as e:
    logger.warning(f"⚠️ Sector caps module import failed: {e} — using inline sector enforcement")

# ── Trading Gate (Circuit Breaker) ──────────────────────────────────
_trading_gate_available = False
try:
    from src.risk.trading_gate import check_trading_allowed, update_breaker_state
    _trading_gate_available = True
    logger.info("✅ Trading gate loaded (circuit breakers)")
except Exception as e:
    logger.warning(f"⚠️ Trading gate import failed: {e} — using inline daily loss check")

# ── Process Lock ────────────────────────────────────────────────────
_process_lock_available = False
try:
    from src.risk.process_lock import acquire_trading_lock, release_trading_lock
    _process_lock_available = True
    logger.info("✅ Process lock loaded")
except Exception as e:
    logger.warning(f"⚠️ Process lock import failed: {e} — skipping lock")

# ── IV Analysis Engine ──────────────────────────────────────────────
_iv_engine = None
try:
    from src.iv_analysis import IVAnalysisEngine, IVMetrics
    _iv_engine = IVAnalysisEngine()
    logger.info("✅ IV analysis engine loaded (5-level fallback)")
except Exception as e:
    logger.warning(f"⚠️ IV analysis import failed: {e} — options IV filtering disabled")

# ── Alpaca Options Engine ───────────────────────────────────────────
_options_engine = None
try:
    from src.alpaca_options_engine import AlpacaOptionsEngine, OptionContract, OptionsPosition
    _options_engine = AlpacaOptionsEngine()
    logger.info("✅ Alpaca options engine loaded")
except Exception as e:
    logger.warning(f"⚠️ Options engine import failed: {e} — options trading disabled")

# ── Enhanced ML Retrainer ───────────────────────────────────────────
_ml_retrainer = None
try:
    from src.ml_retraining_enhanced import EnhancedMLRetrainer, TradeOutcome, RetrainingMetrics
    _ml_retrainer = EnhancedMLRetrainer(
        lookback_days=252,
        retrain_frequency_hours=24,
        use_profit_weighting=True,
        use_regime_conditioning=True,
    )
    logger.info("✅ Enhanced ML retrainer loaded (profit-weighted + regime-aware)")
except Exception as e:
    logger.warning(f"⚠️ Enhanced ML retrainer import failed: {e} — continuous learning disabled")


# ============================================================================
# SECTOR ENFORCEMENT — always active, never falls back to False
# ============================================================================

def get_sector(symbol: str) -> str:
    """Get sector for a symbol. Uses src module if available, else inline map."""
    if _sector_module_available:
        result = _src_get_sector(symbol)
        if result != "unknown":
            return result
    return SECTOR_MAP.get(symbol, "unknown")


def sector_allows_trade_check(
    symbol: str,
    proposed_cost: float,
    current_positions: Dict[str, float],
    total_equity: float,
) -> Tuple[bool, str]:
    """
    Check sector cap. Max 25% per sector, max 2 positions per sector.
    NON-OPTIONAL — always enforced.
    """
    sector = get_sector(symbol)
    max_dollars = total_equity * SECTOR_MAX_PCT

    # Count existing sector exposure and position count
    sector_exposure = 0.0
    sector_count = 0
    for sym, val in current_positions.items():
        if get_sector(sym) == sector:
            sector_exposure += abs(val)
            sector_count += 1

    # Check position count per sector
    if sector_count >= SECTOR_MAX_POSITIONS:
        reason = f"Sector '{sector}' already has {sector_count} positions (max {SECTOR_MAX_POSITIONS})"
        logger.warning(f"🚫 SECTOR CAP: {symbol} blocked — {reason}")
        return False, reason

    # Check dollar exposure per sector
    if sector_exposure + proposed_cost > max_dollars:
        reason = (
            f"Sector '{sector}' exposure would be ${sector_exposure + proposed_cost:,.0f} "
            f"(cap ${max_dollars:,.0f} = {SECTOR_MAX_PCT:.0%} of ${total_equity:,.0f})"
        )
        logger.warning(f"🚫 SECTOR CAP: {symbol} blocked — {reason}")
        return False, reason

    return True, "ok"


# ============================================================================
# REGIME DETECTION — uses Alpaca data, no yfinance
# ============================================================================

class AlpacaRegimeResult:
    """Simple regime result when using Alpaca-only fallback."""
    def __init__(self, regime: str, confidence: float, evidence: dict):
        self.regime = regime
        self.confidence = confidence
        self.evidence = evidence

    @property
    def is_bullish(self) -> bool:
        return self.regime in ("trending_bull", "mean_reverting", "strong_bull", "bull", "neutral")

    @property
    def is_bearish(self) -> bool:
        return self.regime in ("trending_bear", "high_volatility", "bear", "strong_bear", "crisis")


# Regime cache
_regime_cache: Dict[str, Tuple[Any, datetime]] = {}


def detect_regime(symbol: str = "SPY", cfg: UnifiedConfig = None) -> AlpacaRegimeResult:
    """
    Detect market regime. Priority:
      1. src.regime_detector (HMM+GMM) if available
      2. Alpaca-based SMA analysis (fallback)
    
    MANDATORY check — never skipped.
    """
    cfg = cfg or UnifiedConfig()
    cache_key = symbol

    # Check cache
    if cache_key in _regime_cache:
        cached, ts = _regime_cache[cache_key]
        if (datetime.now() - ts).total_seconds() < cfg.regime_cache_minutes * 60:
            return cached

    # === Try src.regime_detector (HMM+GMM) ===
    if _regime_detector is not None:
        try:
            result = _regime_detector.detect_regime(symbol)
            regime_val = result.regime.value if hasattr(result.regime, 'value') else str(result.regime)
            ar = AlpacaRegimeResult(
                regime=regime_val,
                confidence=result.confidence,
                evidence=result.evidence,
            )
            _regime_cache[cache_key] = (ar, datetime.now())
            logger.info(f"Regime [{symbol}]: {ar.regime} (conf={ar.confidence:.0%}) via HMM+GMM")
            return ar
        except Exception as e:
            logger.warning(f"HMM+GMM regime detection failed: {e} — falling back to Alpaca SMA")

    # === Fallback: Alpaca bar data SMA analysis ===
    bars = get_bars(symbol, timeframe="1Day", limit=250)
    if not bars or len(bars) < 50:
        ar = AlpacaRegimeResult("neutral", 0.3, {"error": "insufficient bars"})
        _regime_cache[cache_key] = (ar, datetime.now())
        return ar

    closes = np.array([float(b["c"]) for b in bars])
    price = closes[-1]
    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, min(200, len(closes)))
    sma20 = compute_sma(closes, 20)

    atr = compute_atr(
        np.array([float(b["h"]) for b in bars]),
        np.array([float(b["l"]) for b in bars]),
        closes,
    )
    atr_pct = atr / price if price > 0 else 0.02

    if atr_pct > 0.03:
        regime = "high_volatility"
        confidence = 0.6
    elif price > sma200 and sma50 > sma200 and price > sma50:
        regime = "trending_bull"
        confidence = 0.7
    elif price > sma200:
        regime = "mean_reverting"
        confidence = 0.5
    elif price < sma200 and sma50 < sma200:
        regime = "trending_bear"
        confidence = 0.65
    else:
        regime = "neutral"
        confidence = 0.4

    evidence = {
        "price": f"${price:.2f}",
        "sma20": f"${sma20:.2f}",
        "sma50": f"${sma50:.2f}",
        "sma200": f"${sma200:.2f}",
        "atr_pct": f"{atr_pct:.2%}",
        "source": "alpaca_sma_fallback",
    }

    ar = AlpacaRegimeResult(regime, confidence, evidence)
    _regime_cache[cache_key] = (ar, datetime.now())
    logger.info(f"Regime [{symbol}]: {ar.regime} (conf={ar.confidence:.0%}) via Alpaca SMA")
    return ar


# ============================================================================
# TDA CONFIRMATION
# ============================================================================

def get_tda_score(bars_dict: Dict[str, List[dict]], symbol: str) -> float:
    """
    Get TDA topology alignment score for a symbol.
    Returns float in [-1, 1]: positive = topology supports trade.
    Returns 0.0 (neutral) if TDA unavailable.
    """
    if _tda_engine is None:
        return 0.0  # Neutral — don't block

    try:
        import pandas as pd
        # Build returns DataFrame from multiple symbols' bars
        returns_data = {}
        for sym, bars in bars_dict.items():
            if bars and len(bars) >= 30:
                closes = np.array([float(b["c"]) for b in bars])
                rets = np.diff(np.log(closes))
                returns_data[sym] = rets[-60:]  # Last 60 days

        if len(returns_data) < 5:
            return 0.0

        # Align lengths
        min_len = min(len(v) for v in returns_data.values())
        returns_df = pd.DataFrame({k: v[-min_len:] for k, v in returns_data.items()})

        analysis = _tda_engine.analyze_market(returns_df)
        if analysis is None:
            return 0.0

        turbulence = getattr(analysis, "turbulence_index", 50)
        regime = getattr(analysis, "regime_signal", "RISK_ON")

        # Map to score: low turbulence + RISK_ON = positive, high turbulence = negative
        if regime == "RISK_ON":
            score = float(np.clip(1.0 - turbulence / 100, -1, 1))
        elif regime == "RISK_OFF":
            score = float(np.clip(-turbulence / 100, -1, 0))
        else:
            score = 0.0

        logger.debug(f"TDA [{symbol}]: turbulence={turbulence:.0f}, regime={regime}, score={score:.2f}")
        return score

    except Exception as e:
        logger.debug(f"TDA scoring failed: {e}")
        return 0.0


# ============================================================================
# ML SIGNAL
# ============================================================================

def get_ml_confidence(symbol: str, bars: List[dict]) -> float:
    """
    Get ML model prediction confidence [0, 1].
    Returns 0.5 (neutral) if ML unavailable.
    """
    if _nn_predictor is None:
        return 0.5

    try:
        import pandas as pd
        df = pd.DataFrame([{
            "Open": float(b["o"]),
            "High": float(b["h"]),
            "Low": float(b["l"]),
            "Close": float(b["c"]),
            "Volume": float(b["v"]),
        } for b in bars])

        prediction = _nn_predictor.predict(df)
        if prediction is not None:
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                conf = float(prediction[-1])
            else:
                conf = float(prediction)
            logger.debug(f"ML [{symbol}]: confidence={conf:.3f}")
            return float(np.clip(conf, 0, 1))
    except Exception as e:
        logger.debug(f"ML prediction failed for {symbol}: {e}")

    return 0.5


# ============================================================================
# COMPOSITE SIGNAL — Bayesian-weighted aggregation
# ============================================================================

@dataclass
class CompositeSignal:
    """Final aggregated signal for a symbol."""
    symbol: str
    technical_score: float     # [0, 1]
    regime_score: float        # [0, 1]  (1 = bullish)
    tda_score: float           # [-1, 1]
    ml_confidence: float       # [0, 1]
    composite_score: float     # [0, 1]  weighted combination
    direction: str             # BUY, SELL, HOLD
    confidence: float          # [0, 1]
    atr: float
    atr_pct: float
    price: float
    stop_price: float
    position_size_pct: float
    reasons: List[str] = field(default_factory=list)


def compute_composite_signal(
    symbol: str,
    tech: TechnicalScore,
    regime: AlpacaRegimeResult,
    tda_score: float,
    ml_conf: float,
    cfg: UnifiedConfig,
    equity: float,
    current_positions: Dict[str, float],
    thompson_weights: Optional[Dict[str, float]] = None,
) -> CompositeSignal:
    """
    Combine all signal sources with Bayesian weighting.
    
    Weights (Bayesian-inspired — higher weight to more reliable signals):
      - Technical analysis: 0.35
      - Regime detection:   0.25
      - ML prediction:      0.20
      - TDA topology:       0.20

    If thompson_weights is provided, those override the defaults.
    """
    # Regime score: map regime to [0, 1]
    regime_scores = {
        "trending_bull": 0.9, "strong_bull": 0.9,
        "mean_reverting": 0.65, "bull": 0.75, "neutral": 0.55,
        "high_volatility": 0.3, "trending_bear": 0.15,
        "bear": 0.2, "strong_bear": 0.1, "crisis": 0.05,
    }
    regime_val = regime_scores.get(regime.regime, 0.5)

    # TDA: normalize from [-1,1] to [0,1]
    tda_normalized = (tda_score + 1) / 2.0

    # Bayesian weights — optionally overridden by Thompson Sampling
    if thompson_weights:
        w_tech = thompson_weights.get("technical", 0.35)
        w_regime = thompson_weights.get("regime", 0.25)
        w_ml = thompson_weights.get("ml", 0.20)
        w_tda = thompson_weights.get("tda", 0.20)
    else:
        w_tech = 0.35
        w_regime = 0.25
        w_ml = 0.20
        w_tda = 0.20

    composite = (
        w_tech * tech.score +
        w_regime * regime_val +
        w_ml * ml_conf +
        w_tda * tda_normalized
    )
    composite = float(np.clip(composite, 0, 1))

    # Confidence: agreement between sources
    scores = [tech.score, regime_val, ml_conf, tda_normalized]
    confidence = float(1.0 - np.std(scores))  # Higher agreement = higher confidence

    # Direction
    if composite >= cfg.min_composite_score and regime.is_bullish:
        direction = "BUY"
    elif composite <= 0.35 or regime.is_bearish:
        direction = "SELL"
    else:
        direction = "HOLD"

    # ── ML Hard Filter: reject BUY if ML confidence below threshold ──
    if cfg.ml_hard_filter and direction == "BUY":
        if ml_conf < cfg.ml_min_confidence:
            direction = "HOLD"
            reasons_extra = [f"⛔ ML hard filter: {ml_conf:.2f} < {cfg.ml_min_confidence}"]
        else:
            reasons_extra = []
    else:
        reasons_extra = []

    # ATR-based stop loss
    is_volatile = tech.atr_pct > cfg.volatility_threshold
    atr_mult = cfg.atr_mult_volatile if is_volatile else cfg.atr_mult_stable
    stop_price = tech.price - (tech.atr * atr_mult)

    # Position sizing — Half-Kelly
    position_pct = _compute_position_size(
        composite, confidence, tech.atr_pct, regime, cfg
    )

    # Build reasons
    reasons = []
    if tech.score > 0.6:
        reasons.append(f"Tech bullish ({tech.score:.2f})")
    elif tech.score < 0.4:
        reasons.append(f"Tech bearish ({tech.score:.2f})")
    if regime.is_bullish:
        reasons.append(f"Regime: {regime.regime}")
    else:
        reasons.append(f"⚠️ Regime: {regime.regime}")
    if tda_score > 0.2:
        reasons.append(f"TDA supports ({tda_score:.2f})")
    elif tda_score < -0.2:
        reasons.append(f"TDA warns ({tda_score:.2f})")
    if ml_conf > 0.6:
        reasons.append(f"ML bullish ({ml_conf:.2f})")
    elif ml_conf < 0.4:
        reasons.append(f"ML bearish ({ml_conf:.2f})")
    reasons.extend(reasons_extra)

    return CompositeSignal(
        symbol=symbol,
        technical_score=tech.score,
        regime_score=regime_val,
        tda_score=tda_score,
        ml_confidence=ml_conf,
        composite_score=composite,
        direction=direction,
        confidence=confidence,
        atr=tech.atr,
        atr_pct=tech.atr_pct,
        price=tech.price,
        stop_price=round(stop_price, 2),
        position_size_pct=position_pct,
        reasons=reasons,
    )


def _compute_position_size(
    composite: float,
    confidence: float,
    atr_pct: float,
    regime: AlpacaRegimeResult,
    cfg: UnifiedConfig,
) -> float:
    """
    Half-Kelly position sizing, capped at 5%.
    
    If Kelly sizer available, use it. Otherwise, heuristic sizing
    based on signal strength and volatility.
    """
    if _kelly_sizer is not None:
        try:
            result = _kelly_sizer.calculate_kelly()
            base_pct = result.position_size_pct
        except Exception:
            base_pct = cfg.default_position_pct
    else:
        base_pct = cfg.default_position_pct

    # Scale by composite signal strength
    signal_scale = float(np.clip(composite, 0.5, 1.0))
    base_pct *= signal_scale

    # Scale by inverse volatility (lower size for volatile stocks)
    vol_scale = min(1.0, 0.02 / max(atr_pct, 0.005))
    base_pct *= vol_scale

    # Scale by regime (reduce in bearish/volatile)
    regime_scales = {
        "trending_bull": 1.0, "strong_bull": 1.0, "bull": 0.9,
        "mean_reverting": 0.8, "neutral": 0.7,
        "high_volatility": 0.5, "trending_bear": 0.3,
        "bear": 0.3, "strong_bear": 0.1, "crisis": 0.0,
    }
    regime_scale = regime_scales.get(regime.regime, 0.5)
    base_pct *= regime_scale

    # Clamp to [min, max]
    return float(np.clip(base_pct, cfg.min_position_pct, cfg.max_position_pct))


# ============================================================================
# DAILY LOSS CIRCUIT BREAKER (inline fallback)
# ============================================================================

class InlineCircuitBreaker:
    """Simple daily loss circuit breaker when src.risk.trading_gate unavailable."""

    def __init__(self, max_daily_loss_pct: float = 0.03):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.start_equity: Optional[float] = None
        self.start_date: Optional[date] = None
        self.halted = False

    def reset_daily(self, equity: float):
        today = date.today()
        if self.start_date != today:
            self.start_equity = equity
            self.start_date = today
            self.halted = False

    def check(self, current_equity: float) -> Tuple[bool, str]:
        """Returns (allowed, reason)."""
        if self.start_equity is None:
            self.reset_daily(current_equity)
            return True, "ok"

        self.reset_daily(current_equity)  # Reset if new day

        if self.halted:
            return False, "Circuit breaker halted for today"

        loss_pct = (self.start_equity - current_equity) / self.start_equity
        if loss_pct >= self.max_daily_loss_pct:
            self.halted = True
            logger.error(
                f"🔴 CIRCUIT BREAKER: Daily loss {loss_pct:.2%} >= {self.max_daily_loss_pct:.0%} "
                f"(${self.start_equity:,.0f} → ${current_equity:,.0f}). HALTING."
            )
            return False, f"Daily loss {loss_pct:.2%} exceeds {self.max_daily_loss_pct:.0%} limit"

        return True, "ok"


_inline_breaker = InlineCircuitBreaker()


def check_circuit_breaker(equity: float, confidence: float = 0.5) -> Tuple[bool, str]:
    """Check circuit breaker — uses src module if available, else inline."""
    if _trading_gate_available:
        try:
            allowed, reason = check_trading_allowed(signal_confidence=confidence)
            return allowed, reason
        except Exception as e:
            logger.warning(f"Trading gate check failed: {e}")

    return _inline_breaker.check(equity)


# ============================================================================
# THOMPSON SAMPLING — strategy selection by recent performance
# ============================================================================

class ThompsonSampler:
    """
    Thompson Sampling for strategy/signal-source selection.

    Each "arm" tracks (alpha, beta) of a Beta distribution representing
    the posterior belief about the arm's win probability.

    At decision time, draw a sample from each Beta posterior and pick the
    highest.  Returned weights are normalised to sum to 1.
    """

    def __init__(self, arms: List[str], prior_alpha: float = 1.0,
                 prior_beta: float = 1.0):
        self.arms = arms
        self.alpha: Dict[str, float] = {a: prior_alpha for a in arms}
        self.beta_param: Dict[str, float] = {a: prior_beta for a in arms}

    def update(self, arm: str, success: bool):
        """Record an outcome (win / loss) for *arm*."""
        if arm not in self.alpha:
            self.alpha[arm] = 1.0
            self.beta_param[arm] = 1.0
        if success:
            self.alpha[arm] += 1.0
        else:
            self.beta_param[arm] += 1.0

    def sample_weights(self) -> Dict[str, float]:
        """
        Draw from each posterior and return normalised weights.

        Returns dict[arm_name → weight] summing to 1.
        """
        samples = {}
        for arm in self.arms:
            samples[arm] = float(np.random.beta(self.alpha[arm], self.beta_param[arm]))
        total = sum(samples.values()) or 1.0
        return {a: s / total for a, s in samples.items()}

    def best_arm(self) -> str:
        """Sample once and return the arm with the highest draw."""
        weights = self.sample_weights()
        return max(weights, key=weights.get)

    def stats(self) -> Dict[str, dict]:
        """Return per-arm (alpha, beta, mean)."""
        return {
            a: {
                "alpha": self.alpha[a],
                "beta": self.beta_param[a],
                "mean": self.alpha[a] / (self.alpha[a] + self.beta_param[a]),
            }
            for a in self.arms
        }


# ============================================================================
# OPTIONS SCANNER — iron condors / credit spreads in low-vol regimes
# ============================================================================

@dataclass
class OptionsTradeRecord:
    """Track an open options position."""
    underlying: str
    strategy: str              # "iron_condor", "put_credit_spread", "call_credit_spread"
    entry_time: datetime
    expiration: str            # YYYY-MM-DD
    credit_received: float     # Total net credit ($)
    max_loss: float            # Max risk ($)
    contracts: int
    legs: List[dict]           # [{"symbol": ..., "side": ..., "qty": ...}, ...]
    current_value: float = 0.0
    closed: bool = False
    close_reason: str = ""

    @property
    def pnl(self) -> float:
        """Unrealised P&L (positive = profit for credit trades)."""
        return self.credit_received - self.current_value

    @property
    def pnl_pct_of_credit(self) -> float:
        """P&L as fraction of credit received."""
        if self.credit_received <= 0:
            return 0.0
        return self.pnl / self.credit_received

    @property
    def dte(self) -> int:
        """Days to expiration."""
        try:
            exp = datetime.strptime(self.expiration, "%Y-%m-%d").date()
            return (exp - date.today()).days
        except Exception:
            return 999


# ============================================================================
# POSITION TRACKING
# ============================================================================

@dataclass
class TrackedPosition:
    """Track an open position with stop/target management."""
    symbol: str
    entry_price: float
    entry_time: datetime
    qty: int
    stop_price: float
    target_price: float
    trailing_stop: float
    trailing_active: bool = False
    highest_price: float = 0.0
    atr_at_entry: float = 0.0
    sector: str = ""

    def update_trailing(self, current_price: float, trailing_pct: float = 0.015,
                        activation_pct: float = 0.03):
        """Update trailing stop if price has risen enough."""
        self.highest_price = max(self.highest_price, current_price)
        gain_pct = (current_price - self.entry_price) / self.entry_price

        if gain_pct >= activation_pct:
            self.trailing_active = True
            new_trail = current_price * (1 - trailing_pct)
            self.trailing_stop = max(self.trailing_stop, new_trail)

    @property
    def effective_stop(self) -> float:
        if self.trailing_active:
            return max(self.stop_price, self.trailing_stop)
        return self.stop_price


# ============================================================================
# STATE PERSISTENCE
# ============================================================================

STATE_DIR = Path("state")
STATE_FILE = STATE_DIR / "unified_trader_state.json"


def save_state(positions: Dict[str, TrackedPosition], trade_history: List[dict]):
    """Persist state to JSON for crash recovery."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "timestamp": datetime.now().isoformat(),
        "positions": {
            sym: {
                "symbol": p.symbol, "entry_price": p.entry_price,
                "entry_time": p.entry_time.isoformat(), "qty": p.qty,
                "stop_price": p.stop_price, "target_price": p.target_price,
                "trailing_stop": p.trailing_stop, "trailing_active": p.trailing_active,
                "highest_price": p.highest_price, "atr_at_entry": p.atr_at_entry,
                "sector": p.sector,
            }
            for sym, p in positions.items()
        },
        "recent_trades": trade_history[-50:],
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state() -> Tuple[Dict[str, TrackedPosition], List[dict]]:
    """Load persisted state."""
    positions = {}
    history = []
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            for sym, p in state.get("positions", {}).items():
                positions[sym] = TrackedPosition(
                    symbol=p["symbol"],
                    entry_price=p["entry_price"],
                    entry_time=datetime.fromisoformat(p["entry_time"]),
                    qty=p["qty"],
                    stop_price=p["stop_price"],
                    target_price=p["target_price"],
                    trailing_stop=p.get("trailing_stop", 0),
                    trailing_active=p.get("trailing_active", False),
                    highest_price=p.get("highest_price", p["entry_price"]),
                    atr_at_entry=p.get("atr_at_entry", 0),
                    sector=p.get("sector", ""),
                )
            history = state.get("recent_trades", [])
            logger.info(f"Loaded state: {len(positions)} positions, {len(history)} trades")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    return positions, history


# ============================================================================
# MAIN TRADING ENGINE
# ============================================================================

class UnifiedTrader:
    """
    The ONE production trading engine.
    
    Scan loop:
      1. Check process lock
      2. Check circuit breaker
      3. Detect market regime (MANDATORY)
      4. For each symbol:
         a. Fetch Alpaca bars
         b. Compute technical score
         c. Get TDA topology score
         d. Get ML confidence
         e. Aggregate into composite signal
         f. If BUY: check sector caps, compute Kelly size, submit limit order
         g. If positions exist: check stops, targets, trailing
      5. Options scan (every 30 min in low-vol regime)
      6. Options exit management (every cycle)
      7. Save state
      8. Wait for next scan
    """

    def __init__(self, cfg: UnifiedConfig = None, dry_run: bool = False,
                 scan_only: bool = False):
        self.cfg = cfg or UnifiedConfig()
        self.dry_run = dry_run
        self.scan_only = scan_only
        self.positions: Dict[str, TrackedPosition] = {}
        self.trade_history: List[dict] = []
        self.lock_handle: Optional[IO] = None
        self.running = True
        self.scan_count = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0

        # Kelly trade tracking
        self._win_count = 0
        self._loss_count = 0
        self._total_win_return = 0.0
        self._total_loss_return = 0.0

        # Options tracking
        self.options_positions: List[OptionsTradeRecord] = []
        self._last_options_scan: Optional[datetime] = None

        # Thompson Sampling for signal source selection
        self.thompson = ThompsonSampler(
            arms=["technical", "regime", "ml", "tda"],
            prior_alpha=self.cfg.thompson_prior_alpha,
            prior_beta=self.cfg.thompson_prior_beta,
        ) if self.cfg.thompson_enabled else None

        # Retraining scheduler state
        self._last_retrain_date: Optional[date] = None
        self._retrain_thread: Optional[threading.Thread] = None

        # Load persisted state
        self.positions, self.trade_history = load_state()

        # Signal handlers
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown."""
        logger.info("Shutdown signal received — saving state and exiting")
        self.running = False
        save_state(self.positions, self.trade_history)
        if self.lock_handle and _process_lock_available:
            release_trading_lock(self.lock_handle)
        sys.exit(0)

    # ── Main entry point ────────────────────────────────────────────
    def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("UNIFIED TRADER — Starting")
        logger.info(f"  Universe: {len(UNIVERSE)} symbols")
        logger.info(f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE' if not self.scan_only else 'SCAN ONLY'}")
        logger.info(f"  Max position: {self.cfg.max_position_pct:.0%}")
        logger.info(f"  Kelly fraction: {self.cfg.kelly_fraction}")
        logger.info(f"  ATR period: {self.cfg.atr_period}")
        logger.info(f"  Daily loss halt: {self.cfg.max_daily_loss_pct:.0%}")
        logger.info(f"  Options: {'ENABLED' if self.cfg.options_enabled else 'DISABLED'}")
        logger.info(f"  ML hard filter: {'ON (min={self.cfg.ml_min_confidence})' if self.cfg.ml_hard_filter else 'OFF'}")
        logger.info(f"  Thompson sampling: {'ON' if self.cfg.thompson_enabled else 'OFF'}")
        logger.info(f"  Retraining: {'ON (midnight EST)' if self.cfg.retraining_enabled else 'OFF'}")
        logger.info("=" * 60)

        # Process lock — prevent multiple bots
        if _process_lock_available:
            self.lock_handle = acquire_trading_lock("unified_trader")
            if self.lock_handle is None:
                logger.error("Another trading bot is running! Exiting.")
                sys.exit(1)
        else:
            logger.warning("Process lock unavailable — running without lock protection")

        try:
            while self.running:
                try:
                    self._scan_cycle()
                except KeyboardInterrupt:
                    self._shutdown()
                except Exception as e:
                    logger.error(f"Scan cycle error: {e}\n{traceback.format_exc()}")
                    time.sleep(60)

                if self.scan_only:
                    logger.info("Scan-only mode — exiting after one scan")
                    break

                logger.info(f"Next scan in {self.cfg.scan_interval_sec}s...")
                time.sleep(self.cfg.scan_interval_sec)
        finally:
            save_state(self.positions, self.trade_history)
            if self.lock_handle and _process_lock_available:
                release_trading_lock(self.lock_handle)
            logger.info("Unified trader stopped.")

    # ── Single scan cycle ───────────────────────────────────────────
    def _scan_cycle(self):
        """Execute one full scan cycle."""
        self.scan_count += 1
        logger.info(f"{'─' * 40} Scan #{self.scan_count} {'─' * 40}")

        # 1. Check market open
        if not is_market_open():
            # Even when closed, check if midnight retraining is due
            self._check_retraining()
            logger.info("Market closed — waiting")
            return

        # 2. Get account
        account = get_account()
        if not account:
            logger.error("Cannot get account — skipping scan")
            return

        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        buying_power = float(account.get("buying_power", 0))

        if equity <= 0:
            logger.error(f"Account equity ${equity:,.2f} — cannot trade")
            return

        logger.info(f"Account: equity=${equity:,.2f}  cash=${cash:,.2f}  buying_power=${buying_power:,.2f}")

        # 3. Circuit breaker check
        allowed, reason = check_circuit_breaker(equity)
        if not allowed:
            logger.warning(f"🔴 Circuit breaker: {reason}")
            # Still manage existing positions (stops)
            self._manage_positions(equity)
            return

        # 4. Sync positions with Alpaca
        self._sync_positions()

        # 5. Detect regime (MANDATORY)
        regime = detect_regime("SPY", self.cfg)
        self._last_regime = regime.regime  # Store for ML retrainer feedback
        logger.info(f"Market regime: {regime.regime} (confidence={regime.confidence:.0%})")

        # 6. If bear/crisis regime: tighten stops, skip new longs
        skip_new_longs = False
        if regime.is_bearish:
            logger.warning(f"⚠️ BEARISH regime ({regime.regime}) — skipping new longs, tightening stops")
            skip_new_longs = True
            self._tighten_stops()

        # 7. Manage existing equity positions (stops, targets, trailing)
        self._manage_positions(equity)

        # 8. Options exit management (every cycle)
        if self.cfg.options_enabled:
            self._check_options_exits(equity)

        # 9. Options scan (every N minutes, only in allowed regimes)
        if self.cfg.options_enabled and regime.regime in self.cfg.options_regimes_allowed:
            self._maybe_run_options_scan(equity, regime)

        # 10. Scan for new equity entries (if regime allows)
        if not skip_new_longs and len(self.positions) < self.cfg.max_open_positions:
            self._scan_for_entries(equity, cash, regime)
        elif skip_new_longs:
            logger.info("Skipping new entries — bearish regime")
        else:
            logger.info(f"At max positions ({len(self.positions)}/{self.cfg.max_open_positions})")

        # 11. Check if retraining is due (midnight EST)
        self._check_retraining()

        # 12. Save state
        save_state(self.positions, self.trade_history)

        # 13. Summary
        opts_open = len([o for o in self.options_positions if not o.closed])
        logger.info(
            f"Scan #{self.scan_count} complete: "
            f"{len(self.positions)} equity positions, "
            f"{opts_open} options positions, "
            f"{self.daily_trades} trades today"
        )

    # ── Sync positions with Alpaca ──────────────────────────────────
    def _sync_positions(self):
        """Sync tracked positions with actual Alpaca positions."""
        alpaca_positions = get_positions()
        alpaca_syms = {p["symbol"] for p in alpaca_positions}
        tracked_syms = set(self.positions.keys())

        # Remove tracked positions that no longer exist in Alpaca
        for sym in tracked_syms - alpaca_syms:
            logger.info(f"Position {sym} closed externally — removing from tracker")
            del self.positions[sym]

        # Add Alpaca positions not tracked (e.g., after restart)
        for p in alpaca_positions:
            sym = p["symbol"]
            if sym not in self.positions:
                entry_price = float(p.get("avg_entry_price", 0))
                qty = int(float(p.get("qty", 0)))
                if qty > 0 and entry_price > 0:
                    # Use a default stop based on entry price
                    stop = entry_price * 0.95  # 5% default stop
                    target = entry_price * (1 + self.cfg.profit_target_pct)
                    self.positions[sym] = TrackedPosition(
                        symbol=sym, entry_price=entry_price,
                        entry_time=datetime.now(), qty=qty,
                        stop_price=stop, target_price=target,
                        trailing_stop=0, highest_price=entry_price,
                        sector=get_sector(sym),
                    )
                    logger.info(f"Adopted position: {sym} {qty} @ ${entry_price:.2f}")

    # ── Manage existing positions ───────────────────────────────────
    def _manage_positions(self, equity: float):
        """Check stops, targets, and trailing for all positions."""
        to_close = []

        for sym, pos in self.positions.items():
            price = get_latest_trade(sym)
            if price is None:
                continue

            # Update trailing stop
            pos.update_trailing(price, self.cfg.trailing_stop_pct,
                                self.cfg.trailing_stop_activation)

            effective_stop = pos.effective_stop
            gain_pct = (price - pos.entry_price) / pos.entry_price

            # Check stop loss
            if price <= effective_stop:
                reason = "trailing stop" if pos.trailing_active else "ATR stop"
                logger.warning(
                    f"🛑 STOP HIT [{sym}]: ${price:.2f} <= ${effective_stop:.2f} "
                    f"({reason}, P&L: {gain_pct:+.2%})"
                )
                to_close.append((sym, price, reason))
                continue

            # Check profit target
            if price >= pos.target_price:
                logger.info(
                    f"🎯 TARGET HIT [{sym}]: ${price:.2f} >= ${pos.target_price:.2f} "
                    f"(P&L: {gain_pct:+.2%})"
                )
                to_close.append((sym, price, "profit target"))
                continue

        # Execute closes
        for sym, price, reason in to_close:
            self._close_position(sym, price, reason)

    # ── Tighten stops in bearish regime ─────────────────────────────
    def _tighten_stops(self):
        """Tighten stops by 20% when regime is bearish."""
        for sym, pos in self.positions.items():
            price = get_latest_trade(sym)
            if price and price > 0:
                old_stop = pos.stop_price
                # Move stop up 20% of the distance to current price
                gap = price - old_stop
                new_stop = old_stop + gap * 0.2
                if new_stop > old_stop:
                    pos.stop_price = round(new_stop, 2)
                    logger.info(f"Tightened stop [{sym}]: ${old_stop:.2f} → ${pos.stop_price:.2f}")

    # ── Close position ──────────────────────────────────────────────
    def _close_position(self, symbol: str, price: float, reason: str):
        """Close a position via limit sell order."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        if self.dry_run:
            logger.info(f"[DRY RUN] Would sell {pos.qty} {symbol} @ ${price:.2f} ({reason})")
        else:
            # Limit sell slightly below market
            limit_price = round(price * (1 - self.cfg.limit_buffer_pct), 2)
            result = submit_limit_order(symbol, pos.qty, "sell", limit_price)
            if result:
                logger.info(f"SELL order submitted: {pos.qty} {symbol} @ ${limit_price:.2f} ({reason})")
            else:
                logger.error(f"Failed to submit sell order for {symbol}")
                return

        # Track trade result for Kelly
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        if pnl_pct > 0:
            self._win_count += 1
            self._total_win_return += pnl_pct
        else:
            self._loss_count += 1
            self._total_loss_return += abs(pnl_pct)

        if _kelly_sizer is not None:
            _kelly_sizer.add_trade_result(pnl_pct)

        # Feed trade outcome to Enhanced ML Retrainer
        if _ml_retrainer is not None:
            try:
                outcome = TradeOutcome(
                    timestamp=datetime.now(),
                    ticker=symbol,
                    signal="long",
                    confidence=0.5,
                    prediction=0.5,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    pnl=pnl_pct * pos.entry_price * pos.qty,
                    pnl_pct=pnl_pct,
                    is_closed=True,
                    regime=getattr(self, "_last_regime", "unknown"),
                )
                _ml_retrainer.record_trade_outcome(outcome)
            except Exception as e:
                logger.debug(f"ML retrainer outcome recording failed: {e}")

        # Update Thompson Sampling — reward/penalise each signal source
        if self.thompson is not None:
            won = pnl_pct > 0
            for arm in self.thompson.arms:
                self.thompson.update(arm, won)

        trade_record = {
            "symbol": symbol, "side": "sell", "qty": pos.qty,
            "entry_price": pos.entry_price, "exit_price": price,
            "pnl_pct": round(pnl_pct, 4), "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.trade_history.append(trade_record)
        self.daily_trades += 1

        del self.positions[symbol]
        logger.info(f"Position closed: {symbol} P&L={pnl_pct:+.2%} ({reason})")

    # ── Scan for new entries ────────────────────────────────────────
    def _scan_for_entries(self, equity: float, cash: float, regime: AlpacaRegimeResult):
        """Scan universe for new entry signals."""
        # Get current position values for sector check
        current_pos_values = {}
        for p_data in get_positions():
            sym = p_data["symbol"]
            mktval = abs(float(p_data.get("market_value", 0)))
            current_pos_values[sym] = mktval

        # Thompson Sampling weights (if enabled)
        thompson_weights = None
        if self.thompson is not None:
            thompson_weights = self.thompson.sample_weights()
            logger.debug(f"Thompson weights: {thompson_weights}")

        # Fetch bars for TDA (need multiple symbols)
        all_bars: Dict[str, List[dict]] = {}
        candidates_bars: Dict[str, List[dict]] = {}
        for sym in UNIVERSE:
            if sym in self.positions:
                continue  # Skip if already holding
            bars = get_bars(sym, limit=self.cfg.bars_lookback)
            if bars and len(bars) >= 50:
                candidates_bars[sym] = bars
                all_bars[sym] = bars
            time.sleep(0.15)  # Rate limit

        if not candidates_bars:
            logger.info("No candidates with sufficient bar data")
            return

        # Get TDA score (uses multiple symbols)
        tda_score = get_tda_score(all_bars, "market")

        # Score all candidates
        signals: List[CompositeSignal] = []
        for sym, bars in candidates_bars.items():
            tech = score_technicals(sym, bars, self.cfg)
            if tech is None:
                continue

            ml_conf = get_ml_confidence(sym, bars)

            sig = compute_composite_signal(
                sym, tech, regime, tda_score, ml_conf,
                self.cfg, equity, current_pos_values,
                thompson_weights=thompson_weights,
            )

            if sig.direction == "BUY" and sig.composite_score >= self.cfg.min_composite_score:
                # TDA alignment check
                if sig.tda_score < self.cfg.min_tda_alignment:
                    logger.debug(f"Skipping {sym}: TDA misaligned ({sig.tda_score:.2f})")
                    continue
                signals.append(sig)

        if not signals:
            logger.info("No BUY signals this scan")
            return

        # Sort by composite score (best first)
        signals.sort(key=lambda s: s.composite_score, reverse=True)

        logger.info(f"BUY candidates: {len(signals)}")
        for sig in signals[:5]:
            logger.info(
                f"  {sig.symbol}: composite={sig.composite_score:.3f} "
                f"tech={sig.technical_score:.2f} regime={sig.regime_score:.2f} "
                f"tda={sig.tda_score:.2f} ml={sig.ml_confidence:.2f} "
                f"size={sig.position_size_pct:.1%}"
            )

        # Execute top signals
        for sig in signals:
            if len(self.positions) >= self.cfg.max_open_positions:
                break

            # Sector cap check (NON-OPTIONAL)
            proposed_cost = equity * sig.position_size_pct
            allowed, reason = sector_allows_trade_check(
                sig.symbol, proposed_cost, current_pos_values, equity
            )
            if not allowed:
                continue

            # Check we have enough buying power
            if proposed_cost > cash * 0.95:
                logger.debug(f"Skipping {sig.symbol}: not enough cash (${cash:.0f} < ${proposed_cost:.0f})")
                continue

            # Submit buy order
            qty = int(proposed_cost / sig.price)
            if qty <= 0:
                continue

            # Limit order: last price + buffer
            limit_price = round(sig.price * (1 + self.cfg.limit_buffer_pct), 2)

            if self.dry_run:
                logger.info(
                    f"[DRY RUN] Would BUY {qty} {sig.symbol} @ ${limit_price:.2f} "
                    f"(stop=${sig.stop_price:.2f}, target=${sig.price * (1 + self.cfg.profit_target_pct):.2f})"
                )
            else:
                result = submit_limit_order(sig.symbol, qty, "buy", limit_price)
                if result is None:
                    logger.error(f"Failed to submit buy order for {sig.symbol}")
                    continue
                logger.info(f"BUY order submitted: {qty} {sig.symbol} @ ${limit_price:.2f}")

            # Track position
            target = round(sig.price * (1 + self.cfg.profit_target_pct), 2)
            self.positions[sig.symbol] = TrackedPosition(
                symbol=sig.symbol,
                entry_price=sig.price,
                entry_time=datetime.now(),
                qty=qty,
                stop_price=sig.stop_price,
                target_price=target,
                trailing_stop=0,
                highest_price=sig.price,
                atr_at_entry=sig.atr,
                sector=get_sector(sig.symbol),
            )

            # Update sector exposure tracking
            current_pos_values[sig.symbol] = proposed_cost
            cash -= proposed_cost
            self.daily_trades += 1

            trade_record = {
                "symbol": sig.symbol, "side": "buy", "qty": qty,
                "entry_price": sig.price, "limit_price": limit_price,
                "stop_price": sig.stop_price, "target_price": target,
                "composite_score": sig.composite_score,
                "reasons": sig.reasons,
                "timestamp": datetime.now().isoformat(),
            }
            self.trade_history.append(trade_record)

            logger.info(
                f"✅ ENTERED {sig.symbol}: {qty} shares @ ${sig.price:.2f} "
                f"| stop=${sig.stop_price:.2f} | target=${target:.2f} "
                f"| size={sig.position_size_pct:.1%} "
                f"| {', '.join(sig.reasons)}"
            )

    # ── Options scan (iron condors / credit spreads) ────────────────
    def _maybe_run_options_scan(self, equity: float, regime: AlpacaRegimeResult):
        """Run options scan if enough time has elapsed since last scan."""
        now = datetime.now()
        if self._last_options_scan is not None:
            elapsed = (now - self._last_options_scan).total_seconds() / 60.0
            if elapsed < self.cfg.options_scan_interval_min:
                return
        self._last_options_scan = now
        self._run_options_scan(equity, regime)

    def _run_options_scan(self, equity: float, regime: AlpacaRegimeResult):
        """
        Scan for premium-selling opportunities.

        Strategy selection:
          - MEAN_REVERTING / NEUTRAL → iron condors (delta-neutral)
          - TRENDING_BULL → put credit spreads (bullish)
        
        Guards:
          - IV rank must be > threshold (default 50%)
          - Total options exposure < 20% of equity
          - Per-position max 5% of equity
        """
        if _options_engine is None:
            logger.debug("Options engine unavailable — skipping scan")
            return

        # Check total options exposure
        total_opts_exposure = sum(
            otr.max_loss for otr in self.options_positions if not otr.closed
        )
        max_opts_allowed = equity * self.cfg.options_max_portfolio_pct
        if total_opts_exposure >= max_opts_allowed:
            logger.info(
                f"Options exposure ${total_opts_exposure:,.0f} >= "
                f"cap ${max_opts_allowed:,.0f} — skipping scan"
            )
            return

        logger.info(f"🔍 Options scan — regime={regime.regime}, IV threshold={self.cfg.options_iv_rank_threshold}")

        for underlying in self.cfg.options_underlyings:
            # Already have an options position on this underlying?
            if any(
                otr.underlying == underlying and not otr.closed
                for otr in self.options_positions
            ):
                continue

            # IV rank check via IVAnalysisEngine
            if _iv_engine is not None:
                try:
                    iv_rank = _iv_engine.get_iv_rank(underlying)
                    if iv_rank is None:
                        logger.debug(f"IV rank unavailable for {underlying} — skipping")
                        continue
                    if iv_rank < self.cfg.options_iv_rank_threshold:
                        logger.info(
                            f"  {underlying}: IV rank {iv_rank:.1f}% < "
                            f"{self.cfg.options_iv_rank_threshold}% — skipping"
                        )
                        continue
                    logger.info(f"  {underlying}: IV rank {iv_rank:.1f}% ✅ premium selling candidate")
                except Exception as e:
                    logger.warning(f"IV check failed for {underlying}: {e}")
                    continue
            else:
                logger.debug("IV engine unavailable — proceeding without IV check")

            # Per-position size cap
            per_pos_cap = equity * self.cfg.options_max_per_position_pct

            # Choose strategy based on regime
            if regime.regime in ("mean_reverting", "neutral"):
                strategy = "iron_condor"
            else:
                strategy = "put_credit_spread"

            # Attempt to find and place the trade
            try:
                self._place_options_trade(underlying, strategy, per_pos_cap)
            except Exception as e:
                logger.error(f"Options trade placement failed for {underlying}: {e}")
                continue

    def _place_options_trade(self, underlying: str, strategy: str, max_risk: float):
        """
        Find suitable contracts and place an options trade.

        For now this logs the opportunity (dry or live).  Full MLEG order
        construction uses AlpacaOptionsEngine.get_options_chain() to find
        strikes and submit the spread.
        """
        if _options_engine is None:
            return

        try:
            # Calculate target expiration
            target_exp = (date.today() + timedelta(days=self.cfg.options_target_dte)).isoformat()

            # Fetch chain near target expiration
            chain = _options_engine.get_options_chain(underlying, expiration_date=target_exp)
            if not chain:
                logger.debug(f"No options chain for {underlying} near {target_exp}")
                return

            # Get underlying price
            price = get_latest_trade(underlying)
            if price is None:
                return

            # Filter for ATM options (within 5% of price)
            atm_puts = [c for c in chain if c.option_type == "put"
                        and abs(c.strike - price) / price < 0.05 and c.mid > 0.10]
            atm_calls = [c for c in chain if c.option_type == "call"
                         and abs(c.strike - price) / price < 0.05 and c.mid > 0.10]

            if not atm_puts or not atm_calls:
                logger.debug(f"Insufficient ATM options for {underlying}")
                return

            # Sort by strike
            atm_puts.sort(key=lambda c: c.strike)
            atm_calls.sort(key=lambda c: c.strike)

            if strategy == "put_credit_spread":
                # Sell higher-strike put, buy lower-strike put
                if len(atm_puts) < 2:
                    return
                short_put = atm_puts[-1]  # Higher strike (closer to ATM)
                long_put = atm_puts[0]    # Lower strike (further OTM)
                credit = short_put.mid - long_put.mid
                spread_width = short_put.strike - long_put.strike
                if spread_width <= 0 or credit <= 0:
                    return
                max_loss_per_contract = (spread_width - credit) * 100
                contracts = max(1, int(max_risk / max_loss_per_contract))

                legs = [
                    {"symbol": short_put.symbol, "side": "sell", "qty": contracts},
                    {"symbol": long_put.symbol, "side": "buy", "qty": contracts},
                ]

            elif strategy == "iron_condor":
                # Sell put spread + sell call spread
                if len(atm_puts) < 2 or len(atm_calls) < 2:
                    return
                short_put = atm_puts[-1]
                long_put = atm_puts[0]
                short_call = atm_calls[0]
                long_call = atm_calls[-1]
                put_credit = short_put.mid - long_put.mid
                call_credit = short_call.mid - long_call.mid
                credit = put_credit + call_credit
                put_width = short_put.strike - long_put.strike
                call_width = long_call.strike - short_call.strike
                max_spread = max(put_width, call_width)
                if max_spread <= 0 or credit <= 0:
                    return
                max_loss_per_contract = (max_spread - credit) * 100
                contracts = max(1, int(max_risk / max_loss_per_contract))

                legs = [
                    {"symbol": short_put.symbol, "side": "sell", "qty": contracts},
                    {"symbol": long_put.symbol, "side": "buy", "qty": contracts},
                    {"symbol": short_call.symbol, "side": "sell", "qty": contracts},
                    {"symbol": long_call.symbol, "side": "buy", "qty": contracts},
                ]
            else:
                return

            total_credit = credit * contracts * 100
            total_max_loss = max_loss_per_contract * contracts

            if self.dry_run:
                logger.info(
                    f"[DRY RUN] OPTIONS: {strategy.upper()} {underlying} "
                    f"x{contracts} — credit=${total_credit:,.0f} max_loss=${total_max_loss:,.0f}"
                )
            else:
                # Place individual leg orders (Alpaca paper may not support MLEG)
                for leg in legs:
                    try:
                        _options_engine.place_option_order(
                            symbol=leg["symbol"],
                            qty=leg["qty"],
                            side=leg["side"],
                        )
                    except Exception as e:
                        logger.error(f"Options leg order failed: {leg} — {e}")
                        return
                logger.info(
                    f"✅ OPTIONS: {strategy.upper()} {underlying} "
                    f"x{contracts} — credit=${total_credit:,.0f}"
                )

            # Track the position
            exp_str = atm_puts[0].expiration if atm_puts else target_exp
            self.options_positions.append(OptionsTradeRecord(
                underlying=underlying,
                strategy=strategy,
                entry_time=datetime.now(),
                expiration=exp_str,
                credit_received=total_credit,
                max_loss=total_max_loss,
                contracts=contracts,
                legs=legs,
            ))

        except Exception as e:
            logger.error(f"Options trade construction failed for {underlying}: {e}")

    # ── Options exit management ─────────────────────────────────────
    def _check_options_exits(self, equity: float):
        """
        Manage open options positions: take profit, stop loss, DTE close.

        Exit rules:
          - 50% of max profit → close (take profit)
          - Loss > 2x credit received → close (stop loss)
          - DTE <= 7 → close (time-based)
        """
        if _options_engine is None:
            return

        for otr in self.options_positions:
            if otr.closed:
                continue

            # Update current value from positions
            try:
                positions = _options_engine.get_positions()
                # Sum value of matching legs
                total_val = 0.0
                matched = 0
                for leg in otr.legs:
                    for pos in positions:
                        if pos.symbol == leg["symbol"]:
                            total_val += abs(pos.current_price * pos.quantity * 100)
                            matched += 1
                            break
                if matched > 0:
                    otr.current_value = total_val
            except Exception as e:
                logger.debug(f"Options position value check failed: {e}")
                continue

            should_close = False
            reason = ""

            # Rule 1: Take profit at 50% of credit
            if otr.pnl_pct_of_credit >= self.cfg.options_take_profit_pct:
                should_close = True
                reason = f"TAKE PROFIT ({otr.pnl_pct_of_credit:.0%} of credit)"

            # Rule 2: Stop loss at 2x credit
            elif otr.pnl < 0 and abs(otr.pnl) >= otr.credit_received * self.cfg.options_stop_loss_mult:
                should_close = True
                reason = f"STOP LOSS (loss ${abs(otr.pnl):,.0f} >= {self.cfg.options_stop_loss_mult}x credit)"

            # Rule 3: Close at 7 DTE
            elif otr.dte <= self.cfg.options_min_dte_close:
                should_close = True
                reason = f"DTE close ({otr.dte} DTE remaining)"

            if should_close:
                logger.info(
                    f"📋 Options exit: {otr.strategy} {otr.underlying} — {reason} "
                    f"(P&L: ${otr.pnl:+,.0f})"
                )
                if not self.dry_run:
                    # Close each leg
                    for leg in otr.legs:
                        try:
                            close_side = "buy" if leg["side"] == "sell" else "sell"
                            _options_engine.place_option_order(
                                symbol=leg["symbol"],
                                qty=leg["qty"],
                                side=close_side,
                            )
                        except Exception as e:
                            logger.error(f"Options close leg failed: {leg} — {e}")
                otr.closed = True
                otr.close_reason = reason

    # ── Retraining scheduler ────────────────────────────────────────
    def _check_retraining(self):
        """
        Trigger ML retraining at midnight EST if enabled.

        Runs in a background thread to avoid blocking the scan loop.
        Uses EnhancedMLRetrainer.retrain() with trade outcome feedback.
        """
        if not self.cfg.retraining_enabled:
            return
        if _ml_retrainer is None:
            return

        # Only retrain once per day
        today = date.today()
        if self._last_retrain_date == today:
            return

        # Check if we're past midnight EST
        try:
            import pytz
            est = pytz.timezone("US/Eastern")
            now_est = datetime.now(est)
        except ImportError:
            # Fallback: assume UTC-5
            now_est = datetime.utcnow() - timedelta(hours=5)

        if now_est.hour != self.cfg.retraining_hour_est:
            return

        # Check minimum trade count
        closed_trades = len([t for t in self.trade_history if t.get("side") == "sell"])
        if closed_trades < self.cfg.retraining_min_trades:
            logger.debug(f"Retraining skipped: only {closed_trades} closed trades (need {self.cfg.retraining_min_trades})")
            return

        if self._retrain_thread is not None and self._retrain_thread.is_alive():
            logger.debug("Retraining already in progress")
            return

        logger.info("🔄 Midnight EST — launching ML retraining in background thread")
        self._last_retrain_date = today
        self._retrain_thread = threading.Thread(target=self._run_retraining, daemon=True)
        self._retrain_thread.start()

    def _run_retraining(self):
        """Execute ML retraining (runs in background thread)."""
        try:
            import pandas as pd

            # Collect price data for universe using Alpaca bars
            price_data: Dict[str, Any] = {}
            for sym in UNIVERSE[:10]:  # Retrain on top-10 symbols for speed
                bars = get_bars(sym, limit=300)
                if bars and len(bars) >= 100:
                    df = pd.DataFrame([{
                        "Open": float(b["o"]),
                        "High": float(b["h"]),
                        "Low": float(b["l"]),
                        "Close": float(b["c"]),
                        "Volume": float(b["v"]),
                    } for b in bars])
                    price_data[sym] = df
                time.sleep(0.2)

            if len(price_data) < 3:
                logger.warning("Retraining aborted: insufficient price data")
                return

            metrics = _ml_retrainer.retrain(price_data, epochs=10, validation_split=0.2)
            logger.info(
                f"✅ Retraining complete: {metrics.samples_used} samples, "
                f"accuracy={metrics.profit_weighted_accuracy:.3f}"
            )

        except Exception as e:
            logger.error(f"Retraining failed: {e}\n{traceback.format_exc()}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Production Trader")
    parser.add_argument("--dry-run", action="store_true", help="Log trades without executing")
    parser.add_argument("--scan-only", action="store_true", help="Run one scan and exit")
    parser.add_argument("--interval", type=int, default=300, help="Scan interval in seconds")
    parser.add_argument("--max-positions", type=int, default=12, help="Max open positions")
    args = parser.parse_args()

    cfg = UnifiedConfig(
        scan_interval_sec=args.interval,
        max_open_positions=args.max_positions,
    )

    trader = UnifiedTrader(cfg=cfg, dry_run=args.dry_run, scan_only=args.scan_only)
    trader.run()


if __name__ == "__main__":
    main()
