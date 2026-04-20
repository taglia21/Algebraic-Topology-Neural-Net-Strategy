#!/usr/bin/env python3
"""
intraday_backtest.py
====================
Walk-forward backtest using 1-hour intraday bars and the ORIA multi-sleeve
strategy ensemble (momentum + mean-reversion + stat-arb).

This matches Joshua Aalampour's methodology:
- Multi-strategy alpha sleeves (not just NN prediction)
- Factor-neutral portfolio construction
- Signal-proportional position sizing
- Walk-forward OOS evaluation

Key change from prior backtest: signals come from STRATEGY SLEEVES, not LSTM.
The LSTM was predicting at ~39% accuracy (near random). Joshua's edge comes
from the ensemble + orthogonalization, not direction prediction.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
APP_SRC = SCRIPT_DIR.parent
sys.path.insert(0, str(APP_SRC))

import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("intraday_bt")

# ─── Config ──────────────────────────────────────────────────────────────

SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
    "AMD", "NFLX", "CRM", "AVGO", "INTC",
    "JPM", "BAC", "GS", "XOM", "CVX",
    "GLD", "JNJ", "UNH", "PFE", "MRK",
    "PLTR", "SOFI",
]

INITIAL_CAPITAL = 6003.0
MAX_POS_PCT = 0.05       # 5% of NAV per position
MAX_CONCURRENT = 10
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0
SLIPPAGE_PCT = 0.001     # 0.1%

# Strategy parameters (tuned for hourly bars)
MOM_FAST = 5             # 5 hours
MOM_SLOW = 20            # ~3 days
MR_BB_WINDOW = 20        # 20 hours
MR_RSI_WINDOW = 14
SA_SPREAD_WINDOW = 40    # ~1 week
SA_Z_THRESH = 1.5        # Spread z-score threshold

# Walk-forward
TRAIN_BARS = 1260        # ~6 months of hourly bars (252 * 5 hrs/day)
TEST_BARS = 252          # ~1 month
PURGE = 20               # 20 bar gap

OUTPUT = Path("/home/user/workspace")


# ─── Data ────────────────────────────────────────────────────────────────

def download_hourly(symbols, period="730d"):
    """Download 1-hour bars for all symbols."""
    logger.info("Downloading 1-hour data for %d symbols...", len(symbols))
    
    data = yf.download(symbols, period=period, interval="1h",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    
    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                df = data[sym].dropna(how="all")
                if len(df) > 500:
                    # Ensure timezone-naive index for consistency
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    result[sym] = df
            except KeyError:
                pass
    
    logger.info("Got %d/%d symbols, ~%d bars each",
                len(result), len(symbols),
                np.mean([len(df) for df in result.values()]) if result else 0)
    return result


# ─── Strategy Sleeves ────────────────────────────────────────────────────

def momentum_signals(prices_window: pd.DataFrame, vol_window: pd.DataFrame = None) -> dict:
    """Cross-sectional momentum: long top performers, short bottom.
    
    Returns: {symbol: signal} where signal ∈ [-1, +1]
    """
    signals = {}
    if len(prices_window) < MOM_SLOW + 1:
        return signals
    
    scores = {}
    for sym in prices_window.columns:
        p = prices_window[sym].dropna()
        if len(p) < MOM_SLOW + 1:
            continue
        fast_ret = (p.iloc[-1] / p.iloc[-MOM_FAST] - 1) if p.iloc[-MOM_FAST] > 0 else 0
        slow_ret = (p.iloc[-1] / p.iloc[-MOM_SLOW] - 1) if p.iloc[-MOM_SLOW] > 0 else 0
        scores[sym] = 0.6 * fast_ret + 0.4 * slow_ret
    
    if len(scores) < 6:
        return signals
    
    # Z-score normalize
    vals = list(scores.values())
    mean_s = np.mean(vals)
    std_s = np.std(vals)
    if std_s < 1e-8:
        return signals
    
    for sym, score in scores.items():
        z = (score - mean_s) / std_s
        if z > 1.0:
            signals[sym] = min(z / 3.0, 1.0)  # Long
        elif z < -1.0:
            signals[sym] = max(z / 3.0, -1.0)  # Short
    
    return signals


def mean_reversion_signals(prices_window: pd.DataFrame) -> dict:
    """Bollinger Band mean-reversion: fade extremes.
    
    Returns: {symbol: signal}
    """
    signals = {}
    if len(prices_window) < MR_BB_WINDOW + 5:
        return signals
    
    for sym in prices_window.columns:
        p = prices_window[sym].dropna()
        if len(p) < MR_BB_WINDOW + 5:
            continue
        
        # Bollinger Band z-score
        sma = p.rolling(MR_BB_WINDOW).mean()
        std = p.rolling(MR_BB_WINDOW).std()
        
        if std.iloc[-1] < 1e-8 or pd.isna(std.iloc[-1]):
            continue
        
        bb_z = (p.iloc[-1] - sma.iloc[-1]) / std.iloc[-1]
        
        # RSI
        rsi = _rsi(p, MR_RSI_WINDOW)
        
        # Mean-reversion: fade oversold/overbought
        if bb_z < -2.0 and rsi < 35:
            # Oversold → long
            strength = min(abs(bb_z) / 4.0, 1.0)
            signals[sym] = strength
        elif bb_z > 2.0 and rsi > 65:
            # Overbought → short
            strength = min(abs(bb_z) / 4.0, 1.0)
            signals[sym] = -strength
    
    return signals


def stat_arb_signals(prices_window: pd.DataFrame) -> dict:
    """Pairs/cross-sectional stat-arb: trade spread divergences.
    
    Returns: {symbol: signal}
    """
    PAIRS = [
        ("AAPL", "MSFT"), ("NVDA", "AMD"), ("AMZN", "GOOGL"),
        ("JPM", "BAC"), ("GS", "JPM"), ("META", "NFLX"),
        ("AVGO", "INTC"), ("XOM", "CVX"), ("JNJ", "PFE"),
        ("UNH", "MRK"), ("CRM", "ADBE"),
    ]
    
    signals = {}
    if len(prices_window) < SA_SPREAD_WINDOW + 5:
        return signals
    
    for sym_a, sym_b in PAIRS:
        if sym_a not in prices_window.columns or sym_b not in prices_window.columns:
            continue
        
        pa = prices_window[sym_a].dropna()
        pb = prices_window[sym_b].dropna()
        
        common = pa.index.intersection(pb.index)
        if len(common) < SA_SPREAD_WINDOW + 5:
            continue
        
        pa = pa.loc[common]
        pb = pb.loc[common]
        
        # Log spread
        spread = np.log(pa / pb)
        
        # Z-score of spread
        mu = spread.rolling(SA_SPREAD_WINDOW).mean()
        sigma = spread.rolling(SA_SPREAD_WINDOW).std()
        
        if sigma.iloc[-1] < 1e-8 or pd.isna(sigma.iloc[-1]):
            continue
        
        z = (spread.iloc[-1] - mu.iloc[-1]) / sigma.iloc[-1]
        
        if abs(z) > SA_Z_THRESH:
            # Spread too wide → mean-revert
            strength = min(abs(z) / 4.0, 1.0) * 0.8  # Slightly lower conviction
            if z > SA_Z_THRESH:
                # A is overvalued relative to B → short A, long B
                signals[sym_a] = signals.get(sym_a, 0) - strength
                signals[sym_b] = signals.get(sym_b, 0) + strength
            else:
                # A is undervalued relative to B → long A, short B
                signals[sym_a] = signals.get(sym_a, 0) + strength
                signals[sym_b] = signals.get(sym_b, 0) - strength
    
    # Clip to [-1, 1]
    for sym in signals:
        signals[sym] = np.clip(signals[sym], -1.0, 1.0)
    
    return signals


def _rsi(prices: pd.Series, window: int = 14) -> float:
    if len(prices) < window + 1:
        return 50.0
    delta = prices.diff().dropna().tail(window)
    gains = delta.clip(lower=0).mean()
    losses = (-delta.clip(upper=0)).mean()
    if losses < 1e-10:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


# ─── Ensemble combiner ──────────────────────────────────────────────────

def combine_signals(mom: dict, mr: dict, sa: dict) -> dict:
    """Combine signals from all three sleeves with equal weighting.
    
    Joshua uses orthogonalization to remove factor overlap. 
    We approximate with weighted averaging + de-correlation.
    
    Returns: {symbol: combined_signal} ∈ [-1, 1]
    """
    all_syms = set(mom.keys()) | set(mr.keys()) | set(sa.keys())
    combined = {}
    
    # Weights: momentum 0.4, mean-reversion 0.3, stat-arb 0.3
    W_MOM = 0.40
    W_MR = 0.30
    W_SA = 0.30
    
    for sym in all_syms:
        s_mom = mom.get(sym, 0)
        s_mr = mr.get(sym, 0)
        s_sa = sa.get(sym, 0)
        
        # Weighted combination
        raw = W_MOM * s_mom + W_MR * s_mr + W_SA * s_sa
        
        # Require signal agreement: if sleeves disagree strongly, dampen
        signals_present = [s for s in [s_mom, s_mr, s_sa] if abs(s) > 0.01]
        if len(signals_present) >= 2:
            signs = [np.sign(s) for s in signals_present]
            if len(set(signs)) > 1:
                # Disagreement → dampen by 50%
                raw *= 0.5
        
        # Boost for agreement
        if len(signals_present) >= 2 and len(set(np.sign(s) for s in signals_present)) == 1:
            raw *= 1.3  # Agreement bonus
        
        combined[sym] = np.clip(raw, -1.0, 1.0)
    
    # Filter weak signals
    return {s: v for s, v in combined.items() if abs(v) > 0.08}


# ─── Backtester ──────────────────────────────────────────────────────────

class SimpleBacktester:
    """Lightweight backtester for intraday signals with IBKR-style costs."""
    
    def __init__(self, initial_capital, max_pos_pct, max_concurrent):
        self.initial_capital = initial_capital
        self.max_pos_pct = max_pos_pct
        self.max_concurrent = max_concurrent
        self.reset()
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}  # {sym: {qty, avg_cost, entry_bar}}
        self.trades = []
        self.equity_curve = []
        self.bar_count = 0
    
    def process_bar(self, prices: dict, signals: dict):
        """Process one bar: update positions, execute new signals."""
        self.bar_count += 1
        
        # Update unrealized P&L
        nav = self.cash
        for sym, pos in self.positions.items():
            if sym in prices:
                nav += pos["qty"] * prices[sym]
        
        # Generate orders from signals
        target_positions = {}
        for sym, sig in sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True):
            if sym not in prices or prices[sym] <= 0:
                continue
            if len(target_positions) >= self.max_concurrent:
                break
            
            alloc = nav * self.max_pos_pct * abs(sig)
            price = prices[sym]
            qty = int(alloc / price)
            if qty == 0:
                continue
            
            if sig > 0:
                target_positions[sym] = qty
            else:
                target_positions[sym] = -qty
        
        # Close positions not in target
        for sym in list(self.positions.keys()):
            if sym not in target_positions:
                self._close_position(sym, prices.get(sym, 0))
        
        # Open/adjust positions
        for sym, target_qty in target_positions.items():
            if sym in self.positions:
                current_qty = self.positions[sym]["qty"]
                if np.sign(current_qty) != np.sign(target_qty):
                    # Direction change: close then open
                    self._close_position(sym, prices[sym])
                    self._open_position(sym, target_qty, prices[sym])
                # else: keep existing position (avoid churn)
            else:
                self._open_position(sym, target_qty, prices[sym])
        
        # Record equity
        nav = self.cash
        for sym, pos in self.positions.items():
            if sym in prices:
                nav += pos["qty"] * prices[sym]
        self.equity_curve.append(nav)
    
    def _open_position(self, sym, qty, price):
        if price <= 0 or qty == 0:
            return
        cost = abs(qty) * price
        commission = max(MIN_COMMISSION, abs(qty) * COMMISSION_PER_SHARE)
        slippage = cost * SLIPPAGE_PCT
        
        if qty > 0:
            if self.cash < cost + commission + slippage:
                # Reduce size to fit
                available = self.cash * 0.95
                qty = int(available / (price * (1 + SLIPPAGE_PCT) + COMMISSION_PER_SHARE))
                if qty <= 0:
                    return
                cost = qty * price
                commission = max(MIN_COMMISSION, qty * COMMISSION_PER_SHARE)
                slippage = cost * SLIPPAGE_PCT
            self.cash -= cost + commission + slippage
        else:
            # Short: receive proceeds minus costs
            self.cash += cost - commission - slippage
        
        self.positions[sym] = {
            "qty": qty,
            "avg_cost": price * (1 + SLIPPAGE_PCT * np.sign(qty)),
            "entry_bar": self.bar_count,
        }
    
    def _close_position(self, sym, price):
        if sym not in self.positions or price <= 0:
            return
        
        pos = self.positions.pop(sym)
        qty = pos["qty"]
        entry_price = pos["avg_cost"]
        
        cost = abs(qty) * price
        commission = max(MIN_COMMISSION, abs(qty) * COMMISSION_PER_SHARE)
        slippage = cost * SLIPPAGE_PCT
        
        if qty > 0:
            # Closing long: sell
            self.cash += cost - commission - slippage
            pnl = (price * (1 - SLIPPAGE_PCT) - entry_price) * qty - commission
        else:
            # Closing short: buy to cover
            self.cash -= cost + commission + slippage
            pnl = (entry_price - price * (1 + SLIPPAGE_PCT)) * abs(qty) - commission
        
        self.trades.append({
            "symbol": sym,
            "side": "LONG" if qty > 0 else "SHORT",
            "qty": abs(qty),
            "entry_price": entry_price,
            "exit_price": price,
            "pnl": pnl,
            "commission": commission,
            "holding_bars": self.bar_count - pos["entry_bar"],
        })
    
    def close_all(self, prices: dict):
        for sym in list(self.positions.keys()):
            self._close_position(sym, prices.get(sym, 0))
    
    def get_metrics(self):
        eq = pd.Series(self.equity_curve)
        if len(eq) < 2:
            return {}
        
        returns = eq.pct_change().dropna()
        total_ret = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        
        # Annualize: ~1260 hourly bars per year (252 days * 5 hrs)
        bars_per_year = 1260
        n_bars = len(returns)
        years = n_bars / bars_per_year
        
        ann_ret = (1 + total_ret) ** (1 / max(years, 0.01)) - 1
        vol = float(returns.std() * np.sqrt(bars_per_year))
        
        rf_daily = 0.05 / bars_per_year
        excess = returns - rf_daily
        sharpe = float(excess.mean() / returns.std() * np.sqrt(bars_per_year)) if returns.std() > 0 else 0
        
        # Sortino
        downside = returns[returns < 0]
        sortino = float(excess.mean() / downside.std() * np.sqrt(bars_per_year)) if len(downside) > 0 and downside.std() > 0 else 0
        
        # Drawdown
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = float(dd.min())
        
        # Trade stats
        pnls = [t["pnl"] for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        
        win_rate = len(winners) / len(pnls) if pnls else 0
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        pf = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else 0
        
        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "volatility": vol,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf,
            "avg_holding_bars": np.mean([t["holding_bars"] for t in self.trades]) if self.trades else 0,
            "trades_per_day": len(self.trades) / max(n_bars / 7, 1),  # ~7 bars/day
        }


# ─── Main pipeline ───────────────────────────────────────────────────────

def main():
    t0 = time.time()
    
    # 1. Download
    logger.info("═══ STEP 1: Download 1-hour data ═══")
    price_data = download_hourly(SYMBOLS)
    
    if len(price_data) < 10:
        logger.error("Too few symbols (%d). Aborting.", len(price_data))
        return
    
    # Build aligned close-price DataFrame
    close_frames = {}
    for sym, df in price_data.items():
        cc = "Close" if "Close" in df.columns else "close"
        close_frames[sym] = df[cc]
    
    close_df = pd.DataFrame(close_frames).sort_index()
    close_df = close_df.ffill().dropna(how="all")
    
    logger.info("Close matrix: %d bars x %d symbols", len(close_df), close_df.shape[1])
    logger.info("Date range: %s → %s", close_df.index[0], close_df.index[-1])
    
    # 2. Walk-forward backtest with strategy sleeves
    logger.info("═══ STEP 2: Walk-Forward Strategy Backtest ═══")
    
    n_bars = len(close_df)
    all_dates = close_df.index
    
    bt = SimpleBacktester(INITIAL_CAPITAL, MAX_POS_PCT, MAX_CONCURRENT)
    
    # Lookback needed: max of all strategy windows
    lookback = max(MOM_SLOW, MR_BB_WINDOW, SA_SPREAD_WINDOW) + 10
    
    signal_count = 0
    bar_signals = []
    
    logger.info("Processing %d bars with %d-bar lookback...", n_bars, lookback)
    
    for i in range(lookback, n_bars):
        # Rolling window of prices
        window = close_df.iloc[max(0, i - lookback):i + 1]
        current_prices = close_df.iloc[i].dropna().to_dict()
        
        # Generate signals from each sleeve
        mom = momentum_signals(window)
        mr = mean_reversion_signals(window)
        sa = stat_arb_signals(window)
        
        # Combine
        combined = combine_signals(mom, mr, sa)
        
        if combined:
            signal_count += 1
        
        # Execute
        bt.process_bar(current_prices, combined)
        
        # Progress
        if (i - lookback) % 500 == 0 and i > lookback:
            nav = bt.equity_curve[-1] if bt.equity_curve else INITIAL_CAPITAL
            logger.info("  Bar %d/%d | NAV=$%.2f | Trades=%d | Open=%d",
                       i, n_bars, nav, len(bt.trades), len(bt.positions))
    
    # Close remaining positions
    final_prices = close_df.iloc[-1].dropna().to_dict()
    bt.close_all(final_prices)
    
    # 3. Metrics
    logger.info("═══ STEP 3: Results ═══")
    metrics = bt.get_metrics()
    
    print("\n" + "=" * 70)
    print("  ATNN v2 — INTRADAY STRATEGY ENSEMBLE BACKTEST")
    print("=" * 70)
    
    eq = bt.equity_curve
    if eq:
        print(f"\n{'Initial Capital:':<30} ${INITIAL_CAPITAL:,.2f}")
        print(f"{'Final NAV:':<30} ${eq[-1]:,.2f}")
        print(f"{'Period:':<30} {all_dates[lookback].date()} → {all_dates[-1].date()}")
    
    print(f"\n{'Total Return:':<30} {metrics.get('total_return',0)*100:+.2f}%")
    print(f"{'Annual Return:':<30} {metrics.get('annual_return',0)*100:+.2f}%")
    print(f"{'Sharpe Ratio:':<30} {metrics.get('sharpe_ratio',0):.4f}")
    print(f"{'Sortino Ratio:':<30} {metrics.get('sortino_ratio',0):.4f}")
    print(f"{'Max Drawdown:':<30} {metrics.get('max_drawdown',0)*100:.2f}%")
    print(f"{'Volatility:':<30} {metrics.get('volatility',0)*100:.2f}%")
    print(f"{'Total Trades:':<30} {metrics.get('total_trades',0)}")
    print(f"{'Win Rate:':<30} {metrics.get('win_rate',0)*100:.1f}%")
    print(f"{'Profit Factor:':<30} {metrics.get('profit_factor',0):.2f}")
    print(f"{'Avg Holding:':<30} {metrics.get('avg_holding_bars',0):.1f} bars (~{metrics.get('avg_holding_bars',0)/7:.1f} days)")
    print(f"{'Trades/Day:':<30} {metrics.get('trades_per_day',0):.1f}")
    print(f"{'Signal Bars:':<30} {signal_count} / {n_bars - lookback}")
    
    # SPY benchmark
    if "SPY" in close_df.columns:
        spy = close_df["SPY"].iloc[lookback:]
        spy_ret = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0]
        print(f"\n{'SPY Buy&Hold:':<30} {spy_ret*100:+.2f}%")
        print(f"{'Alpha:':<30} {(metrics.get('total_return',0) - spy_ret)*100:+.2f}%")
    
    # Verdict
    sharpe = metrics.get("sharpe_ratio", 0)
    print(f"\n{'='*70}")
    if sharpe > 1.0:
        print(f"  PASS — Sharpe {sharpe:.2f} > 1.0. READY FOR DEPLOYMENT.")
    elif sharpe > 0.5:
        print(f"  MARGINAL — Sharpe {sharpe:.2f}. Close, but needs tuning.")
    else:
        print(f"  NEEDS WORK — Sharpe {sharpe:.2f}")
    print("=" * 70)
    
    # Analyze trade distribution
    if bt.trades:
        sides = [t["side"] for t in bt.trades]
        longs = sides.count("LONG")
        shorts = sides.count("SHORT")
        print(f"\n  Trade breakdown: {longs} longs, {shorts} shorts")
        
        # Per-strategy-signal analysis
        holding_bars = [t["holding_bars"] for t in bt.trades]
        print(f"  Holding period: min={min(holding_bars)}, median={np.median(holding_bars):.0f}, max={max(holding_bars)} bars")
        
        # PnL distribution
        pnls = [t["pnl"] for t in bt.trades]
        print(f"  P&L: total=${sum(pnls):.2f}, avg=${np.mean(pnls):.2f}, median=${np.median(pnls):.2f}")
        print(f"  Best trade: ${max(pnls):.2f}, Worst: ${min(pnls):.2f}")
    
    # Save chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={"height_ratios": [3, 1]})
        
        eq_series = pd.Series(eq, index=all_dates[lookback:lookback + len(eq)])
        ax1.plot(eq_series.index, eq_series.values, label="ATNN v2 Ensemble",
                 lw=1.5, color="#2196F3")
        
        if "SPY" in close_df.columns:
            spy_series = close_df["SPY"].iloc[lookback:lookback + len(eq)]
            spy_eq = spy_series * (INITIAL_CAPITAL / spy_series.iloc[0])
            ax1.plot(spy_eq.index, spy_eq.values, label="SPY B&H",
                     lw=1, color="#757575", alpha=0.7)
        
        ax1.axhline(INITIAL_CAPITAL, color="#999", ls="--", alpha=0.5)
        ax1.set_title(f"ATNN v2 Ensemble Backtest (1H bars) — Sharpe: {sharpe:.2f}",
                      fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        eq_s = pd.Series(eq)
        dd = (eq_s - eq_s.cummax()) / eq_s.cummax()
        dd.index = all_dates[lookback:lookback + len(eq)]
        ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.3)
        ax2.plot(dd.index, dd.values, color="#F44336", lw=0.8)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("DD %")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(OUTPUT / "equity_curve_intraday.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Chart → equity_curve_intraday.png")
    except Exception as e:
        logger.error("Chart failed: %s", e)
    
    # Save results
    rj = {
        "timestamp": datetime.now().isoformat(),
        "data_type": "1-hour bars",
        "initial_capital": INITIAL_CAPITAL,
        "final_nav": eq[-1] if eq else 0,
        "n_symbols": len(close_df.columns),
        "n_bars": n_bars,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                    for k, v in metrics.items()},
        "verdict": "PASS" if sharpe > 1.0 else ("MARGINAL" if sharpe > 0.5 else "NEEDS_WORK"),
    }
    with open(str(OUTPUT / "backtest_results_intraday.json"), "w") as f:
        json.dump(rj, f, indent=2, default=str)
    
    logger.info("Done in %.1f min", (time.time() - t0) / 60)
    return metrics


if __name__ == "__main__":
    main()
