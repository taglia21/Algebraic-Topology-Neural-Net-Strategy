#!/usr/bin/env python3
"""
intraday_backtest_v3.py — Conviction-based, account-size-aware
================================================================
Key insight: With $6K and IBKR's $1 min commission, we need each
trade to generate >$2 profit. That means:
- Position sizes ~$1000-2000 (16-33% of NAV)
- Only 2-3 positions at a time
- Hold for 2-5 days (35-175 hourly bars)
- Only enter on HIGH conviction signals
- More like swing trading than day trading

This matches reality: Joshua runs $100K. At our scale, we MUST
reduce frequency and increase conviction.
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
logger = logging.getLogger("bt_v3")

SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
    "AMD", "NFLX", "CRM", "AVGO", "INTC",
    "JPM", "BAC", "GS", "XOM", "CVX",
    "GLD", "JNJ", "UNH", "PFE", "MRK",
    "PLTR", "SOFI",
]

CAPITAL = 6003.0
MAX_POS_PCT = 0.25       # 25% per position — concentrated
MAX_POSITIONS = 3        # Only 3 at a time
COMM = 1.0               # $1 per trade (min)
SLIPPAGE = 0.0005

# Conviction thresholds
ENTRY_THRESH = 0.35      # Combined signal must exceed this
EXIT_THRESH = 0.10       # Signal weakened below this → exit
MIN_HOLD = 35            # ~5 trading days minimum
MAX_HOLD = 210           # ~30 trading days maximum

# Stop/target
STOP_LOSS = 0.04         # 4%
TAKE_PROFIT = 0.08       # 8%
TRAILING_STOP = 0.025    # 2.5% trailing from peak

# Strategy lookbacks
MOM_FAST = 14
MOM_SLOW = 70
MR_BB = 40
SA_WINDOW = 70

OUTPUT = Path("/home/user/workspace")


def download_hourly(symbols):
    data = yf.download(symbols, period="730d", interval="1h",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        for s in symbols:
            try:
                df = data[s].dropna(how="all")
                if len(df) > 500:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    result[s] = df
            except KeyError:
                pass
    return result


def momentum_signal(w: pd.DataFrame) -> dict:
    if len(w) < MOM_SLOW + 1:
        return {}
    
    scores = {}
    for s in w.columns:
        p = w[s].dropna()
        if len(p) < MOM_SLOW + 1 or p.iloc[-MOM_FAST] <= 0:
            continue
        fast = p.iloc[-1] / p.iloc[-MOM_FAST] - 1
        slow = p.iloc[-1] / p.iloc[-MOM_SLOW] - 1
        if np.sign(fast) == np.sign(slow):
            scores[s] = 0.5 * fast + 0.5 * slow
    
    if len(scores) < 5:
        return {}
    
    vals = list(scores.values())
    mu, sig = np.mean(vals), np.std(vals)
    if sig < 1e-8:
        return {}
    
    out = {}
    for s, sc in scores.items():
        z = (sc - mu) / sig
        if abs(z) > 1.5:
            out[s] = np.clip(z / 2.5, -1, 1)
    return out


def mean_rev_signal(w: pd.DataFrame) -> dict:
    if len(w) < MR_BB + 5:
        return {}
    out = {}
    for s in w.columns:
        p = w[s].dropna()
        if len(p) < MR_BB + 5:
            continue
        sma = p.rolling(MR_BB).mean().iloc[-1]
        std = p.rolling(MR_BB).std().iloc[-1]
        if std < 1e-8 or pd.isna(std):
            continue
        z = (p.iloc[-1] - sma) / std
        # RSI
        delta = p.diff().dropna().tail(14)
        g = delta.clip(lower=0).mean()
        l = (-delta.clip(upper=0)).mean()
        rsi = 100 - 100 / (1 + g / max(l, 1e-10))
        
        if z < -2.0 and rsi < 30:
            out[s] = min(abs(z) / 3.0, 1.0)
        elif z > 2.0 and rsi > 70:
            out[s] = -min(abs(z) / 3.0, 1.0)
    return out


def stat_arb_signal(w: pd.DataFrame) -> dict:
    PAIRS = [("AAPL","MSFT"),("NVDA","AMD"),("AMZN","GOOGL"),
             ("JPM","BAC"),("GS","JPM"),("META","NFLX"),
             ("AVGO","INTC"),("XOM","CVX"),("JNJ","PFE"),("UNH","MRK")]
    
    if len(w) < SA_WINDOW + 5:
        return {}
    out = {}
    for a, b in PAIRS:
        if a not in w.columns or b not in w.columns:
            continue
        pa, pb = w[a].dropna(), w[b].dropna()
        c = pa.index.intersection(pb.index)
        if len(c) < SA_WINDOW + 5:
            continue
        spread = np.log(pa.loc[c] / pb.loc[c])
        mu = spread.rolling(SA_WINDOW).mean().iloc[-1]
        sig = spread.rolling(SA_WINDOW).std().iloc[-1]
        if sig < 1e-8 or pd.isna(sig):
            continue
        z = (spread.iloc[-1] - mu) / sig
        if abs(z) > 2.0:
            strength = min(abs(z) / 3.5, 0.8)
            if z > 0:
                out[a] = out.get(a, 0) - strength
                out[b] = out.get(b, 0) + strength
            else:
                out[a] = out.get(a, 0) + strength
                out[b] = out.get(b, 0) - strength
    return {s: np.clip(v, -1, 1) for s, v in out.items()}


def ensemble(mom, mr, sa) -> dict:
    """Only return signals above entry threshold."""
    all_s = set(mom) | set(mr) | set(sa)
    out = {}
    for s in all_s:
        m, r, a = mom.get(s, 0), mr.get(s, 0), sa.get(s, 0)
        active = [x for x in [m, r, a] if abs(x) > 0.01]
        if not active:
            continue
        
        raw = 0.4 * m + 0.3 * r + 0.3 * a
        
        if len(active) >= 2:
            signs = set(np.sign(x) for x in active)
            if len(signs) == 1:
                raw *= 1.5  # Strong agreement
            else:
                raw *= 0.3  # Conflict
        
        if abs(raw) >= ENTRY_THRESH:
            out[s] = np.clip(raw, -1, 1)
    return out


class SwingBacktester:
    def __init__(self):
        self.cash = CAPITAL
        self.positions = {}
        self.trades = []
        self.equity = []
        self.bar = 0
    
    def step(self, prices, signals):
        self.bar += 1
        
        # Check stops/targets/max-hold
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            if sym not in prices:
                continue
            
            p = prices[sym]
            entry = pos["cost"]
            peak = pos.get("peak", entry)
            
            if pos["qty"] > 0:
                ret = (p - entry) / entry
                pos["peak"] = max(peak, p)
                trailing_ret = (p - pos["peak"]) / pos["peak"]
            else:
                ret = (entry - p) / entry
                pos["peak"] = min(peak, p) if peak < entry else p
                trailing_ret = (pos["peak"] - p) / pos["peak"] if pos["peak"] > 0 else 0
            
            age = self.bar - pos["entry_bar"]
            
            # Stop loss
            if ret < -STOP_LOSS:
                self._close(sym, p, "stop")
                continue
            
            # Take profit
            if ret > TAKE_PROFIT:
                self._close(sym, p, "target")
                continue
            
            # Trailing stop (only after minimum hold)
            if age >= MIN_HOLD and ret > 0.02 and trailing_ret < -TRAILING_STOP:
                self._close(sym, p, "trail")
                continue
            
            # Max hold
            if age >= MAX_HOLD:
                self._close(sym, p, "max_hold")
                continue
            
            # Signal reversal exit (after min hold)
            if age >= MIN_HOLD:
                sym_signal = signals.get(sym, 0)
                if pos["qty"] > 0 and sym_signal < -EXIT_THRESH:
                    self._close(sym, p, "reversal")
                    continue
                if pos["qty"] < 0 and sym_signal > EXIT_THRESH:
                    self._close(sym, p, "reversal")
                    continue
                # Signal faded
                if abs(sym_signal) < EXIT_THRESH / 2:
                    self._close(sym, p, "fade")
                    continue
        
        # New entries: only top signals, max 3 positions
        sorted_sigs = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for sym, sig in sorted_sigs:
            if sym in self.positions:
                continue
            if len(self.positions) >= MAX_POSITIONS:
                break
            if sym not in prices or prices[sym] <= 0:
                continue
            
            p = prices[sym]
            nav = self._nav(prices)
            alloc = nav * MAX_POS_PCT * min(abs(sig), 1.0)
            qty = int(alloc / p)
            
            if qty == 0:
                continue
            
            # Cost check: need at least 2x commission in expected profit
            expected_hold_return = 0.02  # Expect 2% move over holding period
            expected_profit = qty * p * expected_hold_return
            cost = 2 * COMM  # Round-trip commission
            
            if expected_profit < cost * 2:
                continue
            
            if sig > 0:
                self._open(sym, qty, p)
            else:
                self._open(sym, -qty, p)
        
        self.equity.append(self._nav(prices))
    
    def _nav(self, prices):
        nav = self.cash
        for s, pos in self.positions.items():
            nav += pos["qty"] * prices.get(s, pos["cost"])
        return nav
    
    def _open(self, sym, qty, price):
        if qty == 0:
            return
        cost = abs(qty) * price
        if qty > 0:
            total = cost + COMM + cost * SLIPPAGE
            if self.cash < total:
                qty = int((self.cash * 0.9 - COMM) / (price * (1 + SLIPPAGE)))
                if qty <= 0:
                    return
                cost = qty * price
                total = cost + COMM + cost * SLIPPAGE
            self.cash -= total
        else:
            self.cash += cost - COMM - cost * SLIPPAGE
        
        self.positions[sym] = {"qty": qty, "cost": price, "entry_bar": self.bar, "peak": price}
    
    def _close(self, sym, price, reason=""):
        if sym not in self.positions:
            return
        pos = self.positions.pop(sym)
        qty = pos["qty"]
        cost = abs(qty) * price
        
        if qty > 0:
            self.cash += cost - COMM - cost * SLIPPAGE
            pnl = (price * (1 - SLIPPAGE) - pos["cost"]) * qty - 2 * COMM
        else:
            self.cash -= cost + COMM + cost * SLIPPAGE
            pnl = (pos["cost"] - price * (1 + SLIPPAGE)) * abs(qty) - 2 * COMM
        
        self.trades.append({
            "sym": sym, "side": "L" if qty > 0 else "S",
            "qty": abs(qty), "entry": pos["cost"], "exit": price,
            "pnl": pnl, "hold": self.bar - pos["entry_bar"],
            "reason": reason,
        })
    
    def close_all(self, prices):
        for s in list(self.positions):
            self._close(s, prices.get(s, 0), "eod")
    
    def report(self):
        eq = pd.Series(self.equity)
        if len(eq) < 2:
            return {}
        ret = eq.pct_change().dropna()
        total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        years = len(ret) / 1260
        ann = (1 + total) ** (1 / max(years, 0.01)) - 1
        vol = float(ret.std() * np.sqrt(1260))
        rf = 0.05 / 1260
        sharpe = float((ret.mean() - rf) / ret.std() * np.sqrt(1260)) if ret.std() > 0 else 0
        dd = (eq - eq.cummax()) / eq.cummax()
        
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            "total_return": total, "annual_return": ann,
            "sharpe": sharpe, "max_dd": float(dd.min()),
            "vol": vol, "trades": len(self.trades),
            "win_rate": len(wins)/max(len(pnls),1),
            "pf": abs(sum(wins)/sum(losses)) if losses and sum(losses) else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "avg_hold": np.mean([t["hold"] for t in self.trades]) if self.trades else 0,
            "commission": len(self.trades) * 2 * COMM,
        }


def main():
    t0 = time.time()
    
    logger.info("═══ Download ═══")
    raw = download_hourly(SYMBOLS)
    close = pd.DataFrame({s: df["Close" if "Close" in df.columns else "close"]
                          for s, df in raw.items()}).sort_index().ffill().dropna(how="all")
    logger.info("%d bars x %d symbols", len(close), close.shape[1])
    
    logger.info("═══ Backtest ═══")
    lb = max(MOM_SLOW, MR_BB, SA_WINDOW) + 10
    bt = SwingBacktester()
    
    for i in range(lb, len(close)):
        w = close.iloc[max(0, i-lb-20):i+1]
        prices = close.iloc[i].dropna().to_dict()
        
        mom = momentum_signal(w)
        mr = mean_rev_signal(w)
        sa = stat_arb_signal(w)
        sigs = ensemble(mom, mr, sa)
        
        bt.step(prices, sigs)
        
        if (i-lb) % 500 == 0 and i > lb:
            logger.info("  %d/%d | $%.2f | T=%d | P=%d",
                        i, len(close), bt.equity[-1] if bt.equity else CAPITAL,
                        len(bt.trades), len(bt.positions))
    
    bt.close_all(close.iloc[-1].dropna().to_dict())
    
    logger.info("═══ Report ═══")
    m = bt.report()
    eq = bt.equity
    dates = close.index[lb:lb+len(eq)]
    
    print("\n" + "="*70)
    print("  ATNN v2 — SWING TRADING BACKTEST (1H)")
    print("="*70)
    print(f"\n  ${CAPITAL:,.0f} → ${eq[-1]:,.2f}" if eq else "")
    print(f"  {dates[0].date()} → {dates[-1].date()}" if len(dates) > 0 else "")
    print(f"\n  Return:      {m.get('total_return',0)*100:+.2f}%")
    print(f"  Annual:      {m.get('annual_return',0)*100:+.2f}%")
    print(f"  Sharpe:      {m.get('sharpe',0):.4f}")
    print(f"  Max DD:      {m.get('max_dd',0)*100:.2f}%")
    print(f"  Vol:         {m.get('vol',0)*100:.2f}%")
    print(f"  Trades:      {m.get('trades',0)}")
    print(f"  Win Rate:    {m.get('win_rate',0)*100:.1f}%")
    print(f"  PF:          {m.get('pf',0):.2f}")
    print(f"  Avg Win:     ${m.get('avg_win',0):.2f}")
    print(f"  Avg Loss:    ${m.get('avg_loss',0):.2f}")
    print(f"  Avg Hold:    {m.get('avg_hold',0):.0f} bars (~{m.get('avg_hold',0)/7:.1f} days)")
    print(f"  Commission:  ${m.get('commission',0):.0f}")
    
    if "SPY" in close.columns:
        spy = close["SPY"].iloc[lb:lb+len(eq)]
        spy_r = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0]
        print(f"\n  SPY B&H:     {spy_r*100:+.2f}%")
        print(f"  Alpha:       {(m.get('total_return',0) - spy_r)*100:+.2f}%")
    
    # Exit reason breakdown
    if bt.trades:
        reasons = {}
        for t in bt.trades:
            r = t.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"\n  Exit reasons: {reasons}")
        
        pnls = [t["pnl"] for t in bt.trades]
        print(f"  P&L: total=${sum(pnls):.2f} best=${max(pnls):.2f} worst=${min(pnls):.2f}")
    
    s = m.get("sharpe", 0)
    print(f"\n  {'PASS' if s > 1 else 'MARGINAL' if s > 0.5 else 'NEEDS WORK'} — Sharpe {s:.2f}")
    print("="*70)
    
    # Chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
        eq_s = pd.Series(eq, index=dates[:len(eq)])
        a1.plot(eq_s.index, eq_s.values, label="ATNN v2 Swing", lw=1.5, color="#2196F3")
        if "SPY" in close.columns:
            sp = close["SPY"].iloc[lb:lb+len(eq)]
            a1.plot(sp.index, sp*(CAPITAL/sp.iloc[0]), label="SPY", lw=1, color="#757575", alpha=.7)
        a1.axhline(CAPITAL, color="#999", ls="--", alpha=.5)
        a1.set_title(f"ATNN v2 Swing — Sharpe {s:.2f}", fontsize=14, fontweight="bold")
        a1.set_ylabel("$"); a1.legend(); a1.grid(True, alpha=.3)
        dd = (eq_s - eq_s.cummax()) / eq_s.cummax()
        a2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=.3)
        a2.set_ylabel("DD"); a2.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(str(OUTPUT / "equity_curve_v3.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.error("Chart: %s", e)
    
    with open(str(OUTPUT / "backtest_v3.json"), "w") as f:
        json.dump({"metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v) for k, v in m.items()}}, f, indent=2, default=str)
    
    logger.info("Done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
