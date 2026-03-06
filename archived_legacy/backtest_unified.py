#!/usr/bin/env python3
"""
backtest_unified.py — Streamlined Phase-4 component backtest.

Exercises ALL 5 institutional-grade augmentations:
  1. KellyVolSizer          (position sizing)
  2. LiveRegimeDetector      (regime-aware scaling)
  3. CrossAssetCorrelationMonitor  (correlation gating)
  4. PortfolioGreeksAggregator     (equity beta-delta)
  5. SentimentSignalProcessor      (sentiment boost)

Universe : SPY QQQ AAPL MSFT NVDA GOOGL  (+VIX TLT for cross-asset)
Period   : 1 year daily bars via yfinance
Sim      : day-by-day equity curve with signal generation + sizing

Run:
    source .venv-1/bin/activate
    python backtest_unified.py
"""
from __future__ import annotations

import json, logging, os, sys, time, warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── yfinance ──────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

# ── Project components ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from risk_guardian import KellyVolSizer, SizingResult
from src.regime_detector import LiveRegimeDetector, LiveRegime, RegimeAdjustments
from src.correlation_manager import CrossAssetCorrelationMonitor
from src.greeks_manager import PortfolioGreeksAggregator
from src.sentiment_alpha import SentimentSignalProcessor, SentimentItem

# ======================================================================
# CONFIG
# ======================================================================
TRADE_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL"]
CROSS_ASSETS   = ["^VIX", "TLT"]  # yfinance tickers for cross-asset
INITIAL_EQUITY  = 100_000.0
COST_BPS        = 5          # 5 bps per side
LOOKBACK_WARMUP = 60         # days before first trade
RSI_PERIOD      = 14
MOM_PERIOD      = 20         # momentum look-back
MR_Z_ENTRY      = 1.5        # mean-reversion z-score entry
MR_Z_EXIT       = 0.3
MOM_THRESHOLD   = 0.02       # 2% trailing momentum for entry
MAX_POSITIONS   = 4

# ======================================================================
# DATA
# ======================================================================

def fetch_data(period: str = "1y") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, volumes) DataFrames; columns = symbol."""
    all_tickers = TRADE_UNIVERSE + CROSS_ASSETS
    print(f"  Downloading {all_tickers}  period={period} …")
    raw = yf.download(all_tickers, period=period, progress=False, group_by="ticker")

    prices, volumes = {}, {}
    for tk in all_tickers:
        clean = tk.replace("^", "")  # ^VIX → VIX
        try:
            col = raw[tk] if tk in raw.columns.get_level_values(0) else raw[tk.replace("^", "")]
            s = col["Close"].dropna()
            if len(s) > 30:
                prices[clean] = s
                volumes[clean] = col["Volume"].reindex(s.index).fillna(0)
        except Exception:
            print(f"  ⚠ skipping {tk}")

    pdf = pd.DataFrame(prices).dropna()
    vdf = pd.DataFrame(volumes).reindex(pdf.index).fillna(0)
    print(f"  {len(pdf)} trading days, {len(pdf.columns)} symbols loaded")
    return pdf, vdf


# ======================================================================
# SIMPLE SIGNAL GENERATION
# ======================================================================

def compute_rsi(prices: np.ndarray, period: int = RSI_PERIOD) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    ups = np.clip(deltas, 0, None)
    downs = -np.clip(deltas, None, 0)
    avg_up = ups.mean() or 1e-9
    avg_dn = downs.mean() or 1e-9
    rs = avg_up / avg_dn
    return 100.0 - 100.0 / (1.0 + rs)


def compute_zscore(prices: np.ndarray, window: int = MOM_PERIOD) -> float:
    if len(prices) < window:
        return 0.0
    seg = prices[-window:]
    mu, sigma = seg.mean(), seg.std()
    if sigma < 1e-9:
        return 0.0
    return (prices[-1] - mu) / sigma


def momentum_return(prices: np.ndarray, window: int = MOM_PERIOD) -> float:
    if len(prices) < window + 1:
        return 0.0
    return prices[-1] / prices[-window] - 1.0


@dataclass
class Signal:
    symbol: str
    direction: str        # LONG / SHORT / FLAT
    strategy: str         # momentum / mean_reversion
    confidence: float     # [0, 1]
    entry_price: float
    rsi: float
    z_score: float


def generate_signals(
    prices_df: pd.DataFrame,
    day_idx: int,
    current_positions: Dict[str, int],
    regime: RegimeAdjustments,
) -> List[Signal]:
    """Simple momentum + mean-reversion signal generator."""
    signals: List[Signal] = []

    for sym in TRADE_UNIVERSE:
        if sym not in prices_df.columns:
            continue
        hist = prices_df[sym].values[:day_idx + 1]
        if len(hist) < LOOKBACK_WARMUP:
            continue

        rsi = compute_rsi(hist)
        z   = compute_zscore(hist)
        mom = momentum_return(hist)
        price = float(hist[-1])
        held = sym in current_positions

        # ── Momentum strategy ──
        mom_weight = regime.strategy_weights.get("momentum", 0.15)
        if not held and mom > MOM_THRESHOLD and rsi < 70:
            conf = min(0.4 + abs(mom) * 2, 0.90) * mom_weight / 0.5
            signals.append(Signal(sym, "LONG", "momentum", conf, price, rsi, z))
        elif held and mom < -MOM_THRESHOLD:
            signals.append(Signal(sym, "FLAT", "momentum", 0.7, price, rsi, z))

        # ── Mean-reversion strategy ──
        mr_weight = regime.strategy_weights.get("mr", 0.35)
        if not held and z < -MR_Z_ENTRY and rsi < 35:
            conf = min(0.5 + abs(z) * 0.15, 0.90) * mr_weight / 0.35
            signals.append(Signal(sym, "LONG", "mean_reversion", conf, price, rsi, z))
        elif held and abs(z) < MR_Z_EXIT:
            signals.append(Signal(sym, "FLAT", "mean_reversion", 0.6, price, rsi, z))

    return signals


# ======================================================================
# BACKTEST ENGINE
# ======================================================================

@dataclass
class Trade:
    symbol: str
    direction: str
    strategy: str
    entry_day: int
    entry_price: float
    shares: int
    exit_day: int = -1
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


def run_backtest() -> Dict:
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  Phase-4 Unified Backtest — All 5 Components")
    print("=" * 60)

    # ── 1. Fetch data ─────────────────────────────────────────
    prices_df, vol_df = fetch_data("1y")
    n_days = len(prices_df)

    # ── 2. Initialise components ──────────────────────────────
    kelly       = KellyVolSizer(kelly_fraction=0.5, target_vol=0.12)
    regime_det  = LiveRegimeDetector(lookback_bars=252, refit_days=7)
    corr_mon    = CrossAssetCorrelationMonitor(short_window=21, medium_window=63)
    greeks_agg  = PortfolioGreeksAggregator()
    sentiment   = SentimentSignalProcessor()

    print("  ✓ KellyVolSizer            ready")
    print("  ✓ LiveRegimeDetector        ready")
    print("  ✓ CrossAssetCorrelationMonitor ready")
    print("  ✓ PortfolioGreeksAggregator ready")
    print("  ✓ SentimentSignalProcessor  ready")

    # ── State variables ───────────────────────────────────────
    equity = INITIAL_EQUITY
    cash   = INITIAL_EQUITY
    positions: Dict[str, Tuple[int, float]] = {}  # sym → (shares, entry_price)
    trades: List[Trade] = []
    equity_curve: List[float] = []
    regime_log: List[str] = []
    corr_risk_log: List[float] = []
    delta_log: List[float] = []
    peak_equity = INITIAL_EQUITY
    max_dd = 0.0
    daily_returns: List[float] = []
    component_calls = {
        "kelly_sizing": 0,
        "regime_detect": 0,
        "corr_analysis": 0,
        "greeks_agg": 0,
        "sentiment_proc": 0,
    }
    win_rate_data: List[float] = []  # ongoing trade PnLs for Kelly

    # Preload correlation buffer with first 63 days of returns
    warmup_end = min(LOOKBACK_WARMUP, n_days)
    all_syms = [s for s in TRADE_UNIVERSE + ["VIX", "TLT"] if s in prices_df.columns]
    returns_dict: Dict[str, np.ndarray] = {}
    for sym in all_syms:
        vals = prices_df[sym].values[:warmup_end]
        if len(vals) > 1:
            log_ret = np.diff(np.log(vals))
            returns_dict[sym] = log_ret
    corr_mon.load_returns_matrix(returns_dict)

    print(f"\n  Simulating {n_days - LOOKBACK_WARMUP} trading days …\n")

    # ── DAY-BY-DAY ────────────────────────────────────────────
    for day in range(LOOKBACK_WARMUP, n_days):
        row = prices_df.iloc[day]
        prev_row = prices_df.iloc[day - 1]

        # -- Mark-to-market --
        port_val = cash
        for sym, (shares, _ep) in positions.items():
            if sym in row.index:
                port_val += shares * row[sym]
        equity = port_val
        equity_curve.append(equity)

        # Daily return
        if len(equity_curve) > 1:
            dr = (equity_curve[-1] / equity_curve[-2]) - 1.0
        else:
            dr = 0.0
        daily_returns.append(dr)

        # Max drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_dd:
            max_dd = dd
        current_dd = dd

        # ── Component 1: REGIME DETECTION ─────────────────
        spy_prices = prices_df["SPY"].values[:day + 1] if "SPY" in prices_df else None
        if spy_prices is not None and len(spy_prices) >= 60:
            regime_adj = regime_det.predict_regime(spy_prices)
            component_calls["regime_detect"] += 1
        else:
            regime_adj = RegimeAdjustments(
                regime=LiveRegime.NEUTRAL, confidence=0.5,
                strategy_weights={"pairs": 0.5, "mr": 0.35, "momentum": 0.15},
                position_scale=0.7, stop_multiplier=1.0,
            )
        regime_log.append(regime_adj.regime.value)

        # ── Component 3: CORRELATION MONITORING ───────────
        for sym in all_syms:
            if sym in row.index and sym in prev_row.index:
                p_now, p_prev = row[sym], prev_row[sym]
                if p_prev > 0:
                    corr_mon.update_returns(sym, np.log(p_now / p_prev))

        held_syms = list(positions.keys())
        if held_syms:
            # Force cache invalidation so each day is fresh
            corr_mon._report_cache_ts = None
            corr_report = corr_mon.analyze(held_syms)
            component_calls["corr_analysis"] += 1
            corr_risk_log.append(corr_report.risk_score)
        else:
            corr_risk_log.append(0.0)

        # ── Component 4: GREEKS AGGREGATOR ────────────────
        if positions:
            eq_pos = {
                sym: (sh, float(row[sym]))
                for sym, (sh, _) in positions.items()
                if sym in row.index
            }
            exposure = greeks_agg.aggregate(equity_positions=eq_pos)
            component_calls["greeks_agg"] += 1
            delta_log.append(exposure.total_delta)
        else:
            delta_log.append(0.0)

        # ── Generate signals ──────────────────────────────
        signals = generate_signals(prices_df, day, {s: 1 for s in positions}, regime_adj)

        # ── Process exits first ───────────────────────────
        for sig in signals:
            if sig.direction == "FLAT" and sig.symbol in positions:
                sym = sig.symbol
                shares, ep = positions[sym]
                exit_price = float(row[sym])
                cost = exit_price * shares * COST_BPS / 10_000
                pnl = (exit_price - ep) * shares - cost
                pnl_pct = (exit_price - ep) / ep if ep > 0 else 0.0
                cash += shares * exit_price - cost
                trades.append(Trade(
                    sym, "FLAT", sig.strategy, -1, ep, shares,
                    exit_day=day, exit_price=exit_price, pnl=pnl, pnl_pct=pnl_pct,
                ))
                win_rate_data.append(pnl_pct)
                del positions[sym]

        # ── Process entries ───────────────────────────────
        for sig in [s for s in signals if s.direction == "LONG"]:
            if sig.symbol in positions:
                continue
            if len(positions) >= MAX_POSITIONS:
                continue

            sym = sig.symbol
            price = float(row[sym])

            # ── Component 3b: correlation gating ──────────
            test_syms = held_syms + [sym]
            if len(test_syms) > 1:
                corr_mon._report_cache_ts = None
                blocked, reason = corr_mon.should_block_entry(test_syms)
                component_calls["corr_analysis"] += 1
                if blocked:
                    continue

            # ── Component 5: SENTIMENT BOOST ──────────────
            # Inject synthetic sentiment proportional to momentum
            mom = momentum_return(prices_df[sym].values[:day + 1])
            sent_score = np.clip(mom * 5, -1, 1)  # scale to [-1,1]
            sentiment.inject_items(sym, [SentimentItem(
                text=f"Synthetic backtest signal for {sym}",
                source="backtest", score=sent_score,
                timestamp=datetime.utcnow().isoformat(), relevance=0.8,
            )])
            sent_sig = sentiment.process(sym)
            component_calls["sentiment_proc"] += 1
            boosted_conf = min(sig.confidence + sent_sig.confidence_boost, 0.95)

            # ── Component 2: REGIME SCALING ───────────────
            scaled_conf = boosted_conf * regime_adj.position_scale

            # ── Component 1: KELLY SIZING ─────────────────
            # Compute symbol vol (20-day realized)
            hist_prices = prices_df[sym].values[:day + 1]
            sym_vol = kelly.compute_realized_vol(hist_prices) if hasattr(kelly, 'compute_realized_vol') else 0.20
            port_vol = np.std(daily_returns[-20:]) * np.sqrt(252) if len(daily_returns) >= 20 else 0.10

            # Win/loss stats from completed trades
            wins = [t for t in win_rate_data if t > 0]
            losses = [t for t in win_rate_data if t < 0]
            wr = len(wins) / max(len(win_rate_data), 1) if win_rate_data else 0.55
            avg_w = np.mean(wins) if wins else 0.03
            avg_l = abs(np.mean(losses)) if losses else 0.02

            sizing = kelly.compute_position_size(
                symbol=sym, equity=equity, signal_confidence=scaled_conf,
                symbol_vol=sym_vol, portfolio_vol=port_vol,
                win_rate=wr, avg_win=avg_w, avg_loss=avg_l,
                current_drawdown=current_dd,
            )
            component_calls["kelly_sizing"] += 1

            if sizing.rejected:
                continue

            alloc = sizing.final_size_pct * equity
            shares = max(1, int(alloc / price))
            cost = price * shares * COST_BPS / 10_000
            total_cost = shares * price + cost

            if total_cost > cash:
                shares = max(1, int((cash - cost) / price))
                if shares < 1:
                    continue
                total_cost = shares * price + cost

            cash -= total_cost
            positions[sym] = (shares, price)

    # ── Close remaining positions ─────────────────────────
    final_row = prices_df.iloc[-1]
    for sym, (shares, ep) in list(positions.items()):
        if sym in final_row.index:
            exit_price = float(final_row[sym])
            cost = exit_price * shares * COST_BPS / 10_000
            pnl = (exit_price - ep) * shares - cost
            pnl_pct = (exit_price - ep) / ep if ep > 0 else 0.0
            cash += shares * exit_price - cost
            trades.append(Trade(
                sym, "FLAT", "eod_close", -1, ep, shares,
                exit_day=n_days - 1, exit_price=exit_price, pnl=pnl, pnl_pct=pnl_pct,
            ))
            win_rate_data.append(pnl_pct)
    positions.clear()
    equity = cash
    equity_curve.append(equity)

    # ══════════════════════════════════════════════════════════
    # METRICS
    # ══════════════════════════════════════════════════════════
    total_ret = (equity / INITIAL_EQUITY) - 1.0
    spy_ret = (prices_df["SPY"].iloc[-1] / prices_df["SPY"].iloc[LOOKBACK_WARMUP]) - 1.0 if "SPY" in prices_df else 0.0

    rets = np.array(daily_returns)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-9 else 0.0

    winning = [t for t in trades if t.pnl > 0]
    losing  = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning) / max(len(trades), 1)

    regime_counts = {}
    for r in regime_log:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    elapsed = time.time() - t0

    results = {
        "period": f"{prices_df.index[LOOKBACK_WARMUP].strftime('%Y-%m-%d')} → {prices_df.index[-1].strftime('%Y-%m-%d')}",
        "trading_days": n_days - LOOKBACK_WARMUP,
        "symbols": TRADE_UNIVERSE,
        "performance": {
            "total_return_pct": round(total_ret * 100, 2),
            "spy_return_pct": round(spy_ret * 100, 2),
            "alpha_pct": round((total_ret - spy_ret) * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "final_equity": round(equity, 2),
        },
        "trades": {
            "total": len(trades),
            "winners": len(winning),
            "losers": len(losing),
            "win_rate_pct": round(win_rate * 100, 1),
            "avg_win_pct": round(np.mean([t.pnl_pct for t in winning]) * 100, 2) if winning else 0,
            "avg_loss_pct": round(np.mean([t.pnl_pct for t in losing]) * 100, 2) if losing else 0,
            "total_pnl": round(sum(t.pnl for t in trades), 2),
        },
        "component_calls": component_calls,
        "regime_distribution": regime_counts,
        "avg_corr_risk": round(np.mean(corr_risk_log), 1) if corr_risk_log else 0,
        "avg_portfolio_delta": round(np.mean(delta_log), 3) if delta_log else 0,
        "elapsed_sec": round(elapsed, 1),
    }

    # ── Pretty-print ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    perf = results["performance"]
    trd  = results["trades"]
    print(f"  Period          : {results['period']}")
    print(f"  Trading days    : {results['trading_days']}")
    print(f"  Symbols         : {', '.join(TRADE_UNIVERSE)}")
    print()
    print(f"  Total Return    : {perf['total_return_pct']:+.2f}%")
    print(f"  SPY Return      : {perf['spy_return_pct']:+.2f}%")
    print(f"  Alpha           : {perf['alpha_pct']:+.2f}%")
    print(f"  Sharpe Ratio    : {perf['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown    : {perf['max_drawdown_pct']:.2f}%")
    print(f"  Final Equity    : ${perf['final_equity']:,.2f}")
    print()
    print(f"  Trades          : {trd['total']}")
    print(f"  Winners         : {trd['winners']}  ({trd['win_rate_pct']:.1f}%)")
    print(f"  Losers          : {trd['losers']}")
    print(f"  Avg Win         : {trd['avg_win_pct']:+.2f}%")
    print(f"  Avg Loss        : {trd['avg_loss_pct']:+.2f}%")
    print(f"  Total P&L       : ${trd['total_pnl']:+,.2f}")
    print()
    print("  ── Component Utilisation ──")
    for comp, cnt in component_calls.items():
        tag = "✓" if cnt > 0 else "✗"
        print(f"    {tag} {comp:25s}: {cnt:,} calls")
    print()
    print(f"  Regime distribution : {regime_counts}")
    print(f"  Avg corr risk score : {results['avg_corr_risk']:.1f}/100")
    print(f"  Avg portfolio delta : {results['avg_portfolio_delta']:+.3f}")
    print(f"  Elapsed             : {elapsed:.1f}s")
    print("=" * 60)

    # ── Save JSON ─────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "results", "backtest_unified_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved → {out_path}")

    # ── Equity curve plot (optional) ──────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
        days_idx = range(len(equity_curve))

        # Equity curve
        axes[0].plot(days_idx, equity_curve, label="Strategy", linewidth=1.5)
        spy_eq = (prices_df["SPY"].values[LOOKBACK_WARMUP:] / prices_df["SPY"].values[LOOKBACK_WARMUP]) * INITIAL_EQUITY
        spy_eq = list(spy_eq) + [spy_eq[-1]]  # match length
        axes[0].plot(range(len(spy_eq)), spy_eq[:len(equity_curve)], label="SPY B&H", alpha=0.7, linewidth=1)
        axes[0].set_title("Phase-4 Unified Backtest — Equity Curve")
        axes[0].set_ylabel("Equity ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Regime colour bar
        regime_colours = {"bull": "green", "neutral": "gray", "bear": "red"}
        for i, r in enumerate(regime_log):
            axes[1].axvspan(i, i + 1, color=regime_colours.get(r, "gray"), alpha=0.4)
        axes[1].set_xlim(0, len(regime_log))
        axes[1].set_yticks([])
        axes[1].set_xlabel("Trading Day")
        axes[1].set_title("Regime (green=bull, gray=neutral, red=bear)")

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "results", "backtest_unified_equity.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  Plot  → {plot_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    print()
    return results


# ======================================================================
if __name__ == "__main__":
    run_backtest()
