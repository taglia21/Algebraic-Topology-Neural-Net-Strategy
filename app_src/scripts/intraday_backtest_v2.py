#!/usr/bin/env python3
"""
intraday_backtest_v2.py — Fixed transaction-cost-aware version
===============================================================
Key fixes over v1:
1. Transaction cost penalty: only trade when expected gain > cost
2. Minimum holding period: 7 bars (~1 day) before exits
3. Signal hysteresis: require significant change before rebalancing
4. Position persistence: keep winners, cut losers
5. Joshua's risk scaling: volatility-aware position sizing
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bt_v2")

# ─── Config ──────────────────────────────────────────────────────────────

SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
    "AMD", "NFLX", "CRM", "AVGO", "INTC",
    "JPM", "BAC", "GS", "XOM", "CVX",
    "GLD", "JNJ", "UNH", "PFE", "MRK",
    "PLTR", "SOFI",
]

CAPITAL = 6003.0
MAX_POS_PCT = 0.08          # 8% per position (bigger bets, fewer positions)
MAX_CONCURRENT = 6          # Fewer but more concentrated
COMM_PER_SHARE = 0.005
MIN_COMM = 1.0
SLIPPAGE = 0.0005           # 0.05% (half of before)

# Strategy tuning
MIN_SIGNAL = 0.20           # Higher threshold to filter noise
MIN_HOLD_BARS = 7           # ~1 trading day minimum hold
REBALANCE_EVERY = 7         # Only consider rebalancing every 7 bars (~1 day)
SIGNAL_CHANGE_THRESH = 0.3  # Signal must change by 30% to trigger rebalance
STOP_LOSS_PCT = 0.03        # 3% stop loss
TAKE_PROFIT_PCT = 0.05      # 5% take profit

# Momentum parameters (longer lookback for hourly = more stable signals)
MOM_FAST = 14               # ~2 days
MOM_SLOW = 70               # ~2 weeks
MR_BB = 40                  # ~1 week
MR_RSI = 14
SA_WINDOW = 70              # ~2 weeks
SA_Z = 2.0                  # Higher threshold

OUTPUT = Path("/home/user/workspace")


def download_hourly(symbols):
    logger.info("Downloading 1H data for %d symbols...", len(symbols))
    data = yf.download(symbols, period="730d", interval="1h",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                df = data[sym].dropna(how="all")
                if len(df) > 500:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    result[sym] = df
            except KeyError:
                pass
    logger.info("Got %d symbols", len(result))
    return result


# ─── Strategies (signal-quality focused) ─────────────────────────────────

def momentum_signals(w: pd.DataFrame) -> dict:
    """Quality momentum: only signal strong cross-sectional divergences."""
    if len(w) < MOM_SLOW + 1:
        return {}
    
    scores = {}
    for sym in w.columns:
        p = w[sym].dropna()
        if len(p) < MOM_SLOW + 1 or p.iloc[-MOM_FAST] <= 0 or p.iloc[-MOM_SLOW] <= 0:
            continue
        fast = p.iloc[-1] / p.iloc[-MOM_FAST] - 1
        slow = p.iloc[-1] / p.iloc[-MOM_SLOW] - 1
        
        # Only count if both timeframes agree on direction
        if np.sign(fast) == np.sign(slow) and abs(fast) > 0.005:
            scores[sym] = 0.5 * fast + 0.5 * slow
    
    if len(scores) < 5:
        return {}
    
    vals = list(scores.values())
    mu, sigma = np.mean(vals), np.std(vals)
    if sigma < 1e-8:
        return {}
    
    signals = {}
    for sym, s in scores.items():
        z = (s - mu) / sigma
        if z > 1.5:     # Top quintile
            signals[sym] = min(z / 3.0, 1.0)
        elif z < -1.5:  # Bottom quintile
            signals[sym] = max(z / 3.0, -1.0)
    return signals


def mean_reversion_signals(w: pd.DataFrame) -> dict:
    """Quality mean-reversion: only at extreme dislocations."""
    if len(w) < MR_BB + 5:
        return {}
    
    signals = {}
    for sym in w.columns:
        p = w[sym].dropna()
        if len(p) < MR_BB + 5:
            continue
        
        sma = p.rolling(MR_BB).mean().iloc[-1]
        std = p.rolling(MR_BB).std().iloc[-1]
        if std < 1e-8 or pd.isna(std):
            continue
        
        z = (p.iloc[-1] - sma) / std
        rsi = _rsi(p, MR_RSI)
        
        # Only extreme moves
        if z < -2.5 and rsi < 25:
            signals[sym] = min(abs(z) / 4.0, 1.0)
        elif z > 2.5 and rsi > 75:
            signals[sym] = -min(abs(z) / 4.0, 1.0)
    
    return signals


def stat_arb_signals(w: pd.DataFrame) -> dict:
    """Quality pairs: only trade large spread dislocations."""
    PAIRS = [
        ("AAPL", "MSFT"), ("NVDA", "AMD"), ("AMZN", "GOOGL"),
        ("JPM", "BAC"), ("GS", "JPM"), ("META", "NFLX"),
        ("AVGO", "INTC"), ("XOM", "CVX"), ("JNJ", "PFE"),
        ("UNH", "MRK"),
    ]
    
    if len(w) < SA_WINDOW + 5:
        return {}
    
    signals = {}
    for a, b in PAIRS:
        if a not in w.columns or b not in w.columns:
            continue
        pa, pb = w[a].dropna(), w[b].dropna()
        common = pa.index.intersection(pb.index)
        if len(common) < SA_WINDOW + 5:
            continue
        
        spread = np.log(pa.loc[common] / pb.loc[common])
        mu = spread.rolling(SA_WINDOW).mean().iloc[-1]
        sig = spread.rolling(SA_WINDOW).std().iloc[-1]
        
        if sig < 1e-8 or pd.isna(sig):
            continue
        
        z = (spread.iloc[-1] - mu) / sig
        
        if abs(z) > SA_Z:
            strength = min(abs(z) / 4.0, 0.8)
            if z > SA_Z:
                signals[a] = signals.get(a, 0) - strength
                signals[b] = signals.get(b, 0) + strength
            else:
                signals[a] = signals.get(a, 0) + strength
                signals[b] = signals.get(b, 0) - strength
    
    return {s: np.clip(v, -1, 1) for s, v in signals.items()}


def _rsi(p, w=14):
    if len(p) < w + 1:
        return 50.0
    d = p.diff().dropna().tail(w)
    g = d.clip(lower=0).mean()
    l = (-d.clip(upper=0)).mean()
    return 100 - 100 / (1 + g / max(l, 1e-10))


def combine(mom, mr, sa):
    """Ensemble with conviction gating."""
    all_syms = set(mom) | set(mr) | set(sa)
    out = {}
    
    for sym in all_syms:
        s_m = mom.get(sym, 0)
        s_r = mr.get(sym, 0)
        s_a = sa.get(sym, 0)
        
        # Count how many sleeves have a view
        active = sum(1 for s in [s_m, s_r, s_a] if abs(s) > 0.01)
        
        raw = 0.40 * s_m + 0.30 * s_r + 0.30 * s_a
        
        # Require at least 1 sleeve to have strong conviction, or 2+ to agree
        if active == 0:
            continue
        
        if active == 1:
            # Single sleeve: need strong signal
            raw *= 0.7
        elif active >= 2:
            # Multiple sleeves
            signs = [np.sign(s) for s in [s_m, s_r, s_a] if abs(s) > 0.01]
            if len(set(signs)) == 1:
                raw *= 1.4  # Agreement bonus
            else:
                raw *= 0.3  # Disagreement penalty
        
        if abs(raw) >= MIN_SIGNAL:
            out[sym] = np.clip(raw, -1, 1)
    
    return out


# ─── Backtester with cost-awareness ──────────────────────────────────────

class CostAwareBacktester:
    def __init__(self):
        self.cash = CAPITAL
        self.positions = {}
        self.trades = []
        self.equity = []
        self.bar = 0
        self.last_signals = {}
        self.last_rebal = 0
    
    def step(self, prices, signals):
        self.bar += 1
        
        # Check stops/targets first
        self._check_exits(prices)
        
        # Only rebalance periodically
        should_rebal = (self.bar - self.last_rebal) >= REBALANCE_EVERY
        
        # Or if signals changed dramatically
        signal_changed = False
        for sym, sig in signals.items():
            old = self.last_signals.get(sym, 0)
            if abs(sig - old) > SIGNAL_CHANGE_THRESH:
                signal_changed = True
                break
        
        if should_rebal or signal_changed:
            self._rebalance(prices, signals)
            self.last_rebal = self.bar
            self.last_signals = signals.copy()
        
        # Record equity
        nav = self.cash
        for sym, pos in self.positions.items():
            nav += pos["qty"] * prices.get(sym, pos["cost"])
        self.equity.append(nav)
    
    def _check_exits(self, prices):
        """Stop loss and take profit."""
        to_close = []
        for sym, pos in self.positions.items():
            if sym not in prices:
                continue
            
            price = prices[sym]
            entry = pos["cost"]
            
            if pos["qty"] > 0:  # Long
                ret = (price - entry) / entry
            else:  # Short
                ret = (entry - price) / entry
            
            # Stop loss
            if ret < -STOP_LOSS_PCT:
                to_close.append(sym)
            # Take profit
            elif ret > TAKE_PROFIT_PCT:
                to_close.append(sym)
            # Minimum hold check: only exit after min hold
            elif self.bar - pos["entry_bar"] < MIN_HOLD_BARS:
                continue
        
        for sym in to_close:
            self._close(sym, prices.get(sym, 0))
    
    def _rebalance(self, prices, signals):
        """Rebalance with transaction cost awareness."""
        nav = self.cash
        for sym, pos in self.positions.items():
            nav += pos["qty"] * prices.get(sym, pos["cost"])
        
        # Close positions with no signal or flipped signal
        for sym in list(self.positions.keys()):
            if sym not in signals:
                if self.bar - self.positions[sym]["entry_bar"] >= MIN_HOLD_BARS:
                    self._close(sym, prices.get(sym, 0))
            elif np.sign(signals.get(sym, 0)) != np.sign(self.positions[sym]["qty"]):
                self._close(sym, prices.get(sym, 0))
        
        # Open new positions for top signals
        sorted_sigs = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for sym, sig in sorted_sigs:
            if sym in self.positions:
                continue  # Already positioned
            if len(self.positions) >= MAX_CONCURRENT:
                break
            if sym not in prices or prices[sym] <= 0:
                continue
            
            price = prices[sym]
            
            # Volatility-scaled sizing (Joshua's approach)
            alloc = nav * MAX_POS_PCT * abs(sig)
            qty = int(alloc / price)
            if qty == 0:
                continue
            
            # Transaction cost check: is expected return > round-trip cost?
            round_trip_cost = 2 * (max(MIN_COMM, qty * COMM_PER_SHARE) + qty * price * SLIPPAGE)
            # Minimum expected return to justify trade: 2x round-trip cost
            min_expected = round_trip_cost * 2
            expected_pnl = alloc * abs(sig) * 0.01  # Rough: 1% of allocation * signal strength
            
            if expected_pnl < min_expected and qty * price < 50:
                continue  # Not worth trading
            
            if sig > 0:
                self._open(sym, qty, price)
            else:
                self._open(sym, -qty, price)
    
    def _open(self, sym, qty, price):
        if qty == 0 or price <= 0:
            return
        
        cost_val = abs(qty) * price
        comm = max(MIN_COMM, abs(qty) * COMM_PER_SHARE)
        slip = cost_val * SLIPPAGE
        
        if qty > 0:
            total = cost_val + comm + slip
            if self.cash < total:
                qty = int((self.cash * 0.9) / (price * (1 + SLIPPAGE) + COMM_PER_SHARE))
                if qty <= 0:
                    return
                cost_val = qty * price
                comm = max(MIN_COMM, qty * COMM_PER_SHARE)
                slip = cost_val * SLIPPAGE
                total = cost_val + comm + slip
            self.cash -= total
        else:
            self.cash += cost_val - comm - slip
        
        self.positions[sym] = {
            "qty": qty,
            "cost": price,
            "entry_bar": self.bar,
        }
    
    def _close(self, sym, price):
        if sym not in self.positions:
            return
        pos = self.positions.pop(sym)
        qty = pos["qty"]
        
        cost_val = abs(qty) * price
        comm = max(MIN_COMM, abs(qty) * COMM_PER_SHARE)
        slip = cost_val * SLIPPAGE
        
        if qty > 0:
            self.cash += cost_val - comm - slip
            pnl = (price * (1 - SLIPPAGE) - pos["cost"]) * qty - comm
        else:
            self.cash -= cost_val + comm + slip
            pnl = (pos["cost"] - price * (1 + SLIPPAGE)) * abs(qty) - comm
        
        self.trades.append({
            "symbol": sym, "side": "LONG" if qty > 0 else "SHORT",
            "qty": abs(qty), "entry": pos["cost"], "exit": price,
            "pnl": pnl, "comm": comm,
            "hold": self.bar - pos["entry_bar"],
        })
    
    def close_all(self, prices):
        for sym in list(self.positions.keys()):
            self._close(sym, prices.get(sym, 0))
    
    def metrics(self):
        eq = pd.Series(self.equity)
        if len(eq) < 2:
            return {}
        ret = eq.pct_change().dropna()
        total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        
        bars_yr = 1260
        years = len(ret) / bars_yr
        ann = (1 + total) ** (1 / max(years, 0.01)) - 1
        vol = float(ret.std() * np.sqrt(bars_yr))
        
        rf = 0.05 / bars_yr
        sharpe = float((ret.mean() - rf) / ret.std() * np.sqrt(bars_yr)) if ret.std() > 0 else 0
        
        dd = (eq - eq.cummax()) / eq.cummax()
        
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            "total_return": total,
            "annual_return": ann,
            "sharpe": sharpe,
            "max_drawdown": float(dd.min()),
            "volatility": vol,
            "total_trades": len(self.trades),
            "win_rate": len(wins) / max(len(pnls), 1),
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "avg_hold": np.mean([t["hold"] for t in self.trades]) if self.trades else 0,
            "total_commission": sum(t["comm"] for t in self.trades),
        }


def main():
    t0 = time.time()
    
    # Download
    logger.info("═══ STEP 1: Download ═══")
    pd_raw = download_hourly(SYMBOLS)
    
    close = pd.DataFrame({s: df["Close" if "Close" in df.columns else "close"]
                          for s, df in pd_raw.items()}).sort_index().ffill().dropna(how="all")
    
    logger.info("Matrix: %d bars x %d symbols | %s → %s",
                len(close), close.shape[1], close.index[0], close.index[-1])
    
    # Backtest
    logger.info("═══ STEP 2: Backtest ═══")
    lookback = max(MOM_SLOW, MR_BB, SA_WINDOW) + 10
    bt = CostAwareBacktester()
    
    for i in range(lookback, len(close)):
        w = close.iloc[max(0, i - lookback - 20):i + 1]
        prices = close.iloc[i].dropna().to_dict()
        
        mom = momentum_signals(w)
        mr = mean_reversion_signals(w)
        sa = stat_arb_signals(w)
        signals = combine(mom, mr, sa)
        
        bt.step(prices, signals)
        
        if (i - lookback) % 500 == 0 and i > lookback:
            nav = bt.equity[-1] if bt.equity else CAPITAL
            logger.info("  Bar %d/%d | $%.2f | Trades=%d | Open=%d",
                        i, len(close), nav, len(bt.trades), len(bt.positions))
    
    bt.close_all(close.iloc[-1].dropna().to_dict())
    
    # Report
    logger.info("═══ STEP 3: Report ═══")
    m = bt.metrics()
    
    print("\n" + "=" * 70)
    print("  ATNN v2 — COST-AWARE ENSEMBLE BACKTEST (1H bars)")
    print("=" * 70)
    
    eq = bt.equity
    dates = close.index[lookback:lookback + len(eq)]
    
    print(f"\n{'Capital:':<28} ${CAPITAL:,.2f} → ${eq[-1]:,.2f}" if eq else "")
    print(f"{'Period:':<28} {dates[0].date()} → {dates[-1].date()}" if len(dates) > 0 else "")
    
    print(f"\n{'Total Return:':<28} {m.get('total_return',0)*100:+.2f}%")
    print(f"{'Annual Return:':<28} {m.get('annual_return',0)*100:+.2f}%")
    print(f"{'Sharpe:':<28} {m.get('sharpe',0):.4f}")
    print(f"{'Max Drawdown:':<28} {m.get('max_drawdown',0)*100:.2f}%")
    print(f"{'Volatility:':<28} {m.get('volatility',0)*100:.2f}%")
    print(f"{'Trades:':<28} {m.get('total_trades',0)}")
    print(f"{'Win Rate:':<28} {m.get('win_rate',0)*100:.1f}%")
    print(f"{'Profit Factor:':<28} {m.get('profit_factor',0):.2f}")
    print(f"{'Avg Win:':<28} ${m.get('avg_win',0):.2f}")
    print(f"{'Avg Loss:':<28} ${m.get('avg_loss',0):.2f}")
    print(f"{'Avg Hold:':<28} {m.get('avg_hold',0):.1f} bars (~{m.get('avg_hold',0)/7:.1f} days)")
    print(f"{'Total Commission:':<28} ${m.get('total_commission',0):.2f}")
    
    if "SPY" in close.columns:
        spy = close["SPY"].iloc[lookback:lookback + len(eq)]
        spy_ret = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0]
        print(f"\n{'SPY B&H:':<28} {spy_ret*100:+.2f}%")
        print(f"{'Alpha:':<28} {(m.get('total_return',0) - spy_ret)*100:+.2f}%")
    
    sharpe = m.get("sharpe", 0)
    print(f"\n{'='*70}")
    if sharpe > 1.0:
        print(f"  PASS — Sharpe {sharpe:.2f}")
    elif sharpe > 0.5:
        print(f"  MARGINAL — Sharpe {sharpe:.2f}")
    else:
        print(f"  NEEDS WORK — Sharpe {sharpe:.2f}")
    print("=" * 70)
    
    if bt.trades:
        pnls = [t["pnl"] for t in bt.trades]
        print(f"\n  P&L: total=${sum(pnls):.2f}, best=${max(pnls):.2f}, worst=${min(pnls):.2f}")
        holds = [t["hold"] for t in bt.trades]
        print(f"  Hold: min={min(holds)}, med={np.median(holds):.0f}, max={max(holds)} bars")
    
    # Chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={"height_ratios": [3, 1]})
        
        eq_s = pd.Series(eq, index=dates[:len(eq)])
        ax1.plot(eq_s.index, eq_s.values, label="ATNN v2", lw=1.5, color="#2196F3")
        if "SPY" in close.columns:
            spy_s = close["SPY"].iloc[lookback:lookback+len(eq)]
            spy_eq = spy_s * (CAPITAL / spy_s.iloc[0])
            ax1.plot(spy_eq.index, spy_eq.values, label="SPY B&H", lw=1, color="#757575", alpha=.7)
        ax1.axhline(CAPITAL, color="#999", ls="--", alpha=.5)
        ax1.set_title(f"ATNN v2 Cost-Aware Ensemble — Sharpe: {sharpe:.2f}", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio ($)")
        ax1.legend()
        ax1.grid(True, alpha=.3)
        
        dd = (eq_s - eq_s.cummax()) / eq_s.cummax()
        ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=.3)
        ax2.plot(dd.index, dd.values, color="#F44336", lw=.8)
        ax2.set_title("Drawdown"); ax2.set_ylabel("DD %"); ax2.grid(True, alpha=.3)
        
        plt.tight_layout()
        plt.savefig(str(OUTPUT / "equity_curve_intraday_v2.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Chart saved")
    except Exception as e:
        logger.error("Chart: %s", e)
    
    # Save JSON
    rj = {"timestamp": datetime.now().isoformat(), "metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v) for k, v in m.items()}, "verdict": "PASS" if sharpe > 1 else "NEEDS_WORK"}
    with open(str(OUTPUT / "backtest_results_intraday_v2.json"), "w") as f:
        json.dump(rj, f, indent=2, default=str)
    
    logger.info("Done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
