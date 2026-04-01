#!/usr/bin/env python3
"""
intraday_backtest_v4.py — Daily-rebalance, long-only, minimum-cost
====================================================================
Accepts the constraint: $6K account, $1 min commission, whole shares.

Strategy: daily momentum + mean-reversion screening across 30 symbols.
Enter top 2-3 picks, hold 5-15 days, use trailing stops.
Rebalance once daily (at close). Long-only to avoid short-selling
complications with PDT + small account margin.

Target: beat SPY buy-and-hold by being defensive in drawdowns.
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
logger = logging.getLogger("bt_v4")

SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
    "AMD", "NFLX", "CRM", "AVGO", "INTC",
    "JPM", "BAC", "GS", "XOM", "CVX",
    "GLD", "JNJ", "UNH", "PFE", "MRK",
    "PLTR", "SOFI",
]

CAPITAL = 6003.0
MAX_POS_PCT = 0.30       # 30% per position
MAX_POSITIONS = 3
COMM = 1.0
SLIPPAGE = 0.0003        # 0.03% slippage (limit orders)

# Trade management
STOP_LOSS = 0.06         # 6% stop (wider to avoid noise)
TRAILING_STOP = 0.035    # 3.5% trailing from peak
MIN_HOLD_DAYS = 3
MAX_HOLD_DAYS = 20

OUTPUT = Path("/home/user/workspace")


def download_daily(symbols):
    """Use daily bars (more reliable, cleaner signals)."""
    logger.info("Downloading daily data for %d symbols...", len(symbols))
    data = yf.download(symbols, period="4y", interval="1d",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        for s in symbols:
            try:
                df = data[s].dropna(how="all")
                if len(df) > 200:
                    result[s] = df
            except KeyError:
                pass
    logger.info("Got %d symbols, ~%d bars", len(result),
                np.mean([len(df) for df in result.values()]) if result else 0)
    return result


def compute_rankings(close: pd.DataFrame, volume: pd.DataFrame = None) -> pd.DataFrame:
    """Compute composite ranking score for each symbol each day.
    
    Combines:
    1. Momentum (5d + 20d returns, weighted)
    2. Mean-reversion (Bollinger z-score, RSI)
    3. Volume surge
    4. Relative strength vs SPY
    
    Returns DataFrame with same index/columns as close, values = composite score.
    Higher = more attractive for long.
    """
    scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    
    for sym in close.columns:
        p = close[sym]
        
        # 1. Momentum: 5d and 20d returns
        ret5 = p.pct_change(5)
        ret20 = p.pct_change(20)
        mom_score = 0.6 * ret5 + 0.4 * ret20
        
        # 2. Mean-reversion: BB z-score (contrarian at extremes)
        sma20 = p.rolling(20).mean()
        std20 = p.rolling(20).std()
        bb_z = (p - sma20) / std20.replace(0, np.nan)
        
        # RSI
        delta = p.diff()
        gain = delta.clip(lower=0).ewm(span=14).mean()
        loss = (-delta.clip(upper=0)).ewm(span=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        
        # MR score: oversold = positive, overbought = negative
        mr_score = pd.Series(0.0, index=p.index)
        mr_score[bb_z < -1.5] = 0.3 * abs(bb_z[bb_z < -1.5]) / 3
        mr_score[bb_z > 1.5] = -0.3 * abs(bb_z[bb_z > 1.5]) / 3
        mr_score[(rsi < 30)] += 0.2
        mr_score[(rsi > 70)] -= 0.2
        
        # 3. Relative strength vs SPY (if available)
        rs_score = pd.Series(0.0, index=p.index)
        if "SPY" in close.columns and sym != "SPY":
            spy = close["SPY"]
            rel = (p / spy).pct_change(10)
            rs_score = rel.clip(-0.1, 0.1)
        
        # Composite: momentum-dominant but with MR and RS components
        scores[sym] = 0.50 * mom_score + 0.30 * mr_score + 0.20 * rs_score
    
    return scores


def regime_filter(close: pd.DataFrame) -> pd.Series:
    """Simple regime detection: are we in an uptrend or downtrend?
    
    Returns Series of regime values: 1.0 (bull), 0.5 (neutral), 0.2 (bear)
    This scales position sizes in bear markets.
    """
    if "SPY" not in close.columns:
        return pd.Series(1.0, index=close.index)
    
    spy = close["SPY"]
    sma50 = spy.rolling(50).mean()
    sma200 = spy.rolling(200).mean()
    
    regime = pd.Series(0.5, index=close.index)
    regime[(spy > sma50) & (sma50 > sma200)] = 1.0   # Bull
    regime[(spy < sma50) & (sma50 < sma200)] = 0.2    # Bear
    
    return regime


class DailySwingTrader:
    def __init__(self):
        self.cash = CAPITAL
        self.positions = {}
        self.trades = []
        self.equity = []
        self.day = 0
    
    def process_day(self, prices, scores, regime_scale):
        self.day += 1
        
        # 1. Check exits
        for sym in list(self.positions.keys()):
            if sym not in prices:
                continue
            pos = self.positions[sym]
            p = prices[sym]
            entry = pos["cost"]
            peak = pos.get("peak", entry)
            
            ret = (p - entry) / entry
            pos["peak"] = max(peak, p)
            trail_ret = (p - pos["peak"]) / pos["peak"]
            age = self.day - pos["entry_day"]
            
            # Stop loss
            if ret < -STOP_LOSS:
                self._close(sym, p, "stop")
                continue
            
            # Trailing stop (after min hold, if in profit)
            if age >= MIN_HOLD_DAYS and ret > 0.01 and trail_ret < -TRAILING_STOP:
                self._close(sym, p, "trail")
                continue
            
            # Max hold
            if age >= MAX_HOLD_DAYS:
                self._close(sym, p, "max_hold")
                continue
            
            # Score turned negative (signal faded)
            sym_score = scores.get(sym, 0)
            if age >= MIN_HOLD_DAYS and sym_score < -0.01:
                self._close(sym, p, "score_exit")
                continue
        
        # 2. New entries — pick top-scored symbols not already held
        candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for sym, score in candidates:
            if score < 0.02:  # Minimum score threshold
                break
            if sym in self.positions:
                continue
            if len(self.positions) >= MAX_POSITIONS:
                break
            if sym not in prices or prices[sym] <= 0:
                continue
            
            p = prices[sym]
            nav = self._nav(prices)
            
            # Scale allocation by signal strength AND regime
            alloc = nav * MAX_POS_PCT * min(score * 3, 1.0) * regime_scale
            qty = int(alloc / p)
            
            if qty == 0:
                continue
            
            # Cost check
            trade_value = qty * p
            if trade_value < 100:  # Too small
                continue
            
            self._open(sym, qty, p)
        
        self.equity.append(self._nav(prices))
    
    def _nav(self, prices):
        nav = self.cash
        for s, pos in self.positions.items():
            nav += pos["qty"] * prices.get(s, pos["cost"])
        return nav
    
    def _open(self, sym, qty, price):
        cost = qty * price
        comm = max(COMM, qty * 0.005)
        slip = cost * SLIPPAGE
        total = cost + comm + slip
        
        if self.cash < total:
            qty = int((self.cash * 0.95 - COMM) / (price * (1 + SLIPPAGE)))
            if qty <= 0:
                return
            cost = qty * price
            total = cost + max(COMM, qty * 0.005) + cost * SLIPPAGE
        
        self.cash -= total
        self.positions[sym] = {"qty": qty, "cost": price, "entry_day": self.day, "peak": price}
    
    def _close(self, sym, price, reason=""):
        pos = self.positions.pop(sym)
        qty = pos["qty"]
        proceeds = qty * price
        comm = max(COMM, qty * 0.005)
        slip = proceeds * SLIPPAGE
        
        self.cash += proceeds - comm - slip
        pnl = (price - pos["cost"]) * qty - 2 * max(COMM, qty * 0.005) - qty * price * SLIPPAGE * 2
        
        self.trades.append({
            "sym": sym, "qty": qty,
            "entry": pos["cost"], "exit": price,
            "pnl": pnl, "hold": self.day - pos["entry_day"],
            "reason": reason,
        })
    
    def close_all(self, prices):
        for s in list(self.positions):
            self._close(s, prices.get(s, 0), "eod_final")
    
    def report(self):
        eq = pd.Series(self.equity)
        if len(eq) < 10:
            return {}
        ret = eq.pct_change().dropna()
        total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        years = len(ret) / 252
        ann = (1 + total) ** (1 / max(years, 0.01)) - 1
        vol = float(ret.std() * np.sqrt(252))
        rf = 0.05 / 252
        sharpe = float((ret.mean() - rf) / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
        
        # Sortino
        ds = ret[ret < 0]
        sortino = float((ret.mean() - rf) / ds.std() * np.sqrt(252)) if len(ds) > 0 and ds.std() > 0 else 0
        
        dd = (eq - eq.cummax()) / eq.cummax()
        
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            "total_return": total, "annual_return": ann,
            "sharpe": sharpe, "sortino": sortino,
            "max_dd": float(dd.min()), "vol": vol,
            "trades": len(self.trades),
            "win_rate": len(wins)/max(len(pnls),1),
            "pf": abs(sum(wins)/sum(losses)) if losses and sum(losses) else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "avg_hold": np.mean([t["hold"] for t in self.trades]) if self.trades else 0,
            "commission": sum(max(COMM, t["qty"]*0.005)*2 for t in self.trades),
        }


def main():
    t0 = time.time()
    
    logger.info("═══ Download ═══")
    raw = download_daily(SYMBOLS)
    
    close = pd.DataFrame({s: df["Close" if "Close" in df.columns else "close"]
                          for s, df in raw.items()}).sort_index().ffill()
    close = close.dropna(how="all")
    
    logger.info("%d days x %d symbols | %s → %s",
                len(close), close.shape[1], close.index[0].date(), close.index[-1].date())
    
    logger.info("═══ Score + Backtest ═══")
    scores_df = compute_rankings(close)
    regime = regime_filter(close)
    
    bt = DailySwingTrader()
    warmup = 60  # Need 60 days for indicators
    
    for i in range(warmup, len(close)):
        date = close.index[i]
        prices = close.iloc[i].dropna().to_dict()
        day_scores = scores_df.iloc[i].dropna().to_dict()
        reg = float(regime.iloc[i])
        
        bt.process_day(prices, day_scores, reg)
        
        if (i - warmup) % 100 == 0 and i > warmup:
            logger.info("  Day %d/%d | $%.2f | T=%d | P=%d | Regime=%.1f",
                        i, len(close), bt.equity[-1] if bt.equity else CAPITAL,
                        len(bt.trades), len(bt.positions), reg)
    
    bt.close_all(close.iloc[-1].dropna().to_dict())
    
    # Report
    m = bt.report()
    eq = bt.equity
    dates = close.index[warmup:warmup+len(eq)]
    
    print("\n" + "="*70)
    print("  ATNN v2 — DAILY SWING (LONG-ONLY, REGIME-AWARE)")
    print("="*70)
    print(f"\n  ${CAPITAL:,.0f} → ${eq[-1]:,.2f}" if eq else "")
    print(f"  {dates[0].date()} → {dates[-1].date()}" if len(dates) > 0 else "")
    print(f"\n  Return:      {m.get('total_return',0)*100:+.2f}%")
    print(f"  Annual:      {m.get('annual_return',0)*100:+.2f}%")
    print(f"  Sharpe:      {m.get('sharpe',0):.4f}")
    print(f"  Sortino:     {m.get('sortino',0):.4f}")
    print(f"  Max DD:      {m.get('max_dd',0)*100:.2f}%")
    print(f"  Vol:         {m.get('vol',0)*100:.2f}%")
    print(f"  Trades:      {m.get('trades',0)}")
    print(f"  Win Rate:    {m.get('win_rate',0)*100:.1f}%")
    print(f"  PF:          {m.get('pf',0):.2f}")
    print(f"  Avg Win:     ${m.get('avg_win',0):.2f}")
    print(f"  Avg Loss:    ${m.get('avg_loss',0):.2f}")
    print(f"  Avg Hold:    {m.get('avg_hold',0):.0f} days")
    print(f"  Commission:  ${m.get('commission',0):.0f}")
    
    if "SPY" in close.columns:
        spy = close["SPY"].iloc[warmup:warmup+len(eq)]
        if len(spy) > 1:
            spy_r = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0]
            print(f"\n  SPY B&H:     {spy_r*100:+.2f}%")
            print(f"  Alpha:       {(m.get('total_return',0) - spy_r)*100:+.2f}%")
    
    if bt.trades:
        reasons = {}
        for t in bt.trades:
            r = t.get("reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"\n  Exits: {reasons}")
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
            sp = close["SPY"].iloc[warmup:warmup+len(eq)]
            a1.plot(sp.index, sp*(CAPITAL/sp.iloc[0]), label="SPY B&H", lw=1, color="#757575", alpha=.7)
        a1.axhline(CAPITAL, color="#999", ls="--", alpha=.5)
        a1.set_title(f"ATNN v2 Daily Swing — Sharpe {s:.2f}", fontsize=14, fontweight="bold")
        a1.set_ylabel("$"); a1.legend(); a1.grid(True, alpha=.3)
        dd = (eq_s - eq_s.cummax()) / eq_s.cummax()
        a2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=.3)
        a2.set_ylabel("DD"); a2.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(str(OUTPUT / "equity_curve_v4.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except: pass
    
    with open(str(OUTPUT / "backtest_v4.json"), "w") as f:
        json.dump({"metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v) for k, v in m.items()}}, f, indent=2, default=str)
    
    logger.info("Done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
