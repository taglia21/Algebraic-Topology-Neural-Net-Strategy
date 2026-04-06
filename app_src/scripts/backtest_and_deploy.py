#!/usr/bin/env python3
"""
backtest_and_deploy.py
======================
Validates and deploys a two-strategy system:

Strategy 1 — SPY IBS Mean Reversion (primary alpha source)
  Entry: SPY closes below (10-day high – 2.5 × 25-day avg range) AND IBS < 0.30
  IBS = (Close – Low) / (High – Low)
  Exit: SPY closes above previous day's high
  Source: Reddit r/algotrading 25-year backtest, Sharpe 2.11

Strategy 2 — Dual Momentum GEM (regime filter / cash management)
  Hold SPY if 12-month return > T-bill AND SPY > EFA
  Hold EFA if 12-month EFA return > SPY AND EFA has absolute momentum
  Otherwise hold IEF (intermediate Treasuries)
  Source: Gary Antonacci, OptimalMomentum.com

Run: python scripts/backtest_and_deploy.py
"""

from __future__ import annotations
import json, logging, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backtest")

CAPITAL     = 5923.0
OUTPUT      = Path("/home/user/workspace")

# ─── Data Download ────────────────────────────────────────────────────────────

def download(tickers, period="5y"):
    log.info("Downloading %s...", tickers)
    raw = yf.download(tickers, period=period, interval="1d",
                      group_by="ticker", auto_adjust=True,
                      threads=True, progress=False)
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
                if len(df) > 100:
                    out[t] = df
            except KeyError:
                pass
    else:
        out[tickers[0]] = raw.dropna(how="all")
    return out

# ─── Strategy 1: SPY IBS Mean Reversion ──────────────────────────────────────

def spy_ibs_signals(spy_df: pd.DataFrame,
                    atr_window: int = 25,
                    high_window: int = 10,
                    atr_mult: float = 2.5,
                    ibs_thresh: float = 0.30) -> pd.Series:
    """
    Returns daily signal:  +1 = enter/stay long,  0 = flat
    Entry:  Close < (rolling high_window high - atr_mult * rolling atr_window ATR)
            AND IBS < ibs_thresh
    Exit:   Close > previous day's High  (or stop if -4% below entry)
    """
    h = spy_df["High"]
    l = spy_df["Low"]
    c = spy_df["Close"]

    ibs       = (c - l) / (h - l).replace(0, np.nan)
    daily_rng = h - l
    avg_rng   = daily_rng.rolling(atr_window).mean()
    roll_high = h.rolling(high_window).max()
    threshold = roll_high - atr_mult * avg_rng

    in_pos   = False
    entry_hi = 0.0
    signals  = pd.Series(0, index=spy_df.index)

    for i in range(max(atr_window, high_window) + 1, len(spy_df)):
        prev_high = float(h.iloc[i - 1])
        curr_close = float(c.iloc[i])
        curr_ibs   = float(ibs.iloc[i])
        curr_thr   = float(threshold.iloc[i])

        if in_pos:
            # Exit: close above previous high  OR  hard stop -4%
            stop_hit  = curr_close < entry_hi * 0.96
            exit_hit  = curr_close > prev_high
            if exit_hit or stop_hit:
                in_pos = False
            else:
                signals.iloc[i] = 1          # still holding
        else:
            # Entry check (using TODAY's data — we act at next open)
            if curr_close < curr_thr and curr_ibs < ibs_thresh:
                in_pos   = True
                entry_hi = curr_close
                signals.iloc[i] = 1

    return signals

# ─── Strategy 2: Dual Momentum GEM ───────────────────────────────────────────

def gem_signals(prices: dict[str, pd.DataFrame],
                lookback: int = 252) -> pd.DataFrame:
    """
    Monthly GEM signals.
    Returns DataFrame with 'hold' column: 'SPY' | 'EFA' | 'IEF'
    """
    closes = {t: df["Close"] for t, df in prices.items() if "Close" in df.columns}
    close_df = pd.DataFrame(closes).ffill()

    monthly_closes = close_df.resample("ME").last()
    ret_12m = monthly_closes.pct_change(12)

    tbill_return = 0.04 / 12  # ~4% annual T-bill

    results = []
    for i in range(12, len(monthly_closes)):
        date = monthly_closes.index[i]
        spy_ret = float(ret_12m["SPY"].iloc[i]) if "SPY" in ret_12m else 0
        efa_ret = float(ret_12m["EFA"].iloc[i]) if "EFA" in ret_12m else 0

        # Absolute momentum: beat T-bills?
        spy_abs = spy_ret > tbill_return
        efa_abs = efa_ret > tbill_return

        if not spy_abs and not efa_abs:
            hold = "IEF"   # both in absolute bear — go to bonds
        elif spy_ret >= efa_ret:
            hold = "SPY" if spy_abs else "IEF"
        else:
            hold = "EFA" if efa_abs else "IEF"

        results.append({"date": date, "hold": hold,
                         "spy_ret_12m": round(spy_ret * 100, 2),
                         "efa_ret_12m": round(efa_ret * 100, 2)})

    return pd.DataFrame(results).set_index("date")

# ─── Backtester ───────────────────────────────────────────────────────────────

def backtest_ibs(spy_df: pd.DataFrame, signals: pd.Series) -> dict:
    """Simulate IBS strategy with IBKR Pro costs."""
    cash   = CAPITAL
    shares = 0
    entry  = 0.0
    equity = []
    trades = []

    opens   = spy_df["Open"]
    closes  = spy_df["Close"]

    for i in range(1, len(spy_df)):
        date     = spy_df.index[i]
        prev_sig = int(signals.iloc[i - 1])
        curr_sig = int(signals.iloc[i])
        px_open  = float(opens.iloc[i])
        px_close = float(closes.iloc[i])

        # Execute at next open based on yesterday's signal
        if shares == 0 and prev_sig == 1:
            # Enter: buy at open with full capital
            shares = int(cash / px_open)
            if shares > 0:
                cost = shares * px_open
                comm = max(1.0, shares * 0.005)
                cash -= cost + comm
                entry = px_open

        elif shares > 0 and prev_sig == 0:
            # Exit: sell at open
            proceeds = shares * px_open
            comm = max(1.0, shares * 0.005)
            pnl  = proceeds - comm - (shares * entry + max(1.0, shares * 0.005))
            trades.append({"date": str(date.date()), "pnl": round(pnl, 2),
                            "entry": round(entry, 2), "exit": round(px_open, 2),
                            "win": pnl > 0})
            cash += proceeds - comm
            shares = 0
            entry  = 0.0

        nav = cash + shares * px_close
        equity.append(nav)

    # Close final position
    if shares > 0:
        px = float(closes.iloc[-1])
        proceeds = shares * px
        comm = max(1.0, shares * 0.005)
        pnl  = proceeds - comm - (shares * entry + max(1.0, shares * 0.005))
        trades.append({"pnl": round(pnl, 2), "entry": round(entry, 2),
                        "exit": round(px, 2), "win": pnl > 0})
        cash += proceeds - comm
        equity[-1] = cash

    eq = pd.Series(equity, index=spy_df.index[1:])
    ret = eq.pct_change().dropna()
    total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
    years = len(ret) / 252
    ann   = (1 + total) ** (1 / max(years, 0.01)) - 1
    vol   = float(ret.std() * np.sqrt(252))
    rf    = 0.05 / 252
    sharpe = float((ret.mean() - rf) / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    dd     = (eq - eq.cummax()) / eq.cummax()

    wins   = [t["pnl"] for t in trades if t["win"]]
    losses = [t["pnl"] for t in trades if not t["win"]]

    return {
        "strategy": "SPY_IBS",
        "start": str(eq.index[0].date()),
        "end":   str(eq.index[-1].date()),
        "capital": CAPITAL,
        "final_nav": round(float(eq.iloc[-1]), 2),
        "total_return": round(total * 100, 2),
        "annual_return": round(ann * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(float(dd.min()) * 100, 2),
        "volatility": round(vol * 100, 2),
        "total_trades": len(trades),
        "win_rate": round(len(wins) / max(len(trades), 1) * 100, 1),
        "profit_factor": round(abs(sum(wins) / sum(losses)) if losses and sum(losses) else 0, 2),
        "avg_win":  round(np.mean(wins)   if wins   else 0, 2),
        "avg_loss": round(np.mean(losses) if losses else 0, 2),
        "trades": trades,
        "_equity": eq,
        "_spy_bh": (spy_df["Close"].iloc[1:] / spy_df["Close"].iloc[1]) * CAPITAL,
    }


def backtest_gem(prices: dict, gem_df: pd.DataFrame) -> dict:
    """Simulate GEM with monthly rebalancing and IBKR costs."""
    close = {t: df["Close"] for t, df in prices.items() if "Close" in df.columns}
    daily = pd.DataFrame(close).ffill()

    cash   = CAPITAL
    shares = 0
    holding = None
    equity  = []
    trades  = []

    for date in daily.index:
        # Check monthly rebalance
        month_end_dates = gem_df.index
        nearest = month_end_dates[month_end_dates <= date]
        if len(nearest) > 0:
            target = gem_df.loc[nearest[-1], "hold"]
        else:
            target = "IEF"

        if target != holding:
            # Sell current
            if holding and holding in daily.columns and shares > 0:
                px = float(daily.loc[date, holding])
                proceeds = shares * px
                comm = max(1.0, shares * 0.005)
                pnl  = proceeds - comm
                cash += pnl
                trades.append({"from": holding, "to": target, "date": str(date.date())})
                shares = 0

            # Buy new
            if target in daily.columns:
                px = float(daily.loc[date, target])
                shares = int(cash / px)
                if shares > 0:
                    cost = shares * px
                    comm = max(1.0, shares * 0.005)
                    cash -= cost + comm
            holding = target

        # NAV
        nav = cash
        if holding and holding in daily.columns and shares > 0:
            nav += shares * float(daily.loc[date, holding])
        equity.append(nav)

    eq  = pd.Series(equity, index=daily.index)
    ret = eq.pct_change().dropna()
    total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
    years = len(ret) / 252
    ann   = (1 + total) ** (1 / max(years, 0.01)) - 1
    sharpe = float((ret.mean() - 0.05/252) / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    dd     = (eq - eq.cummax()) / eq.cummax()

    return {
        "strategy": "GEM",
        "final_nav": round(float(eq.iloc[-1]), 2),
        "total_return": round(total * 100, 2),
        "annual_return": round(ann * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(float(dd.min()) * 100, 2),
        "trades": len(trades),
        "_equity": eq,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("ATNN — Evidence-Based Strategy Backtest")
    log.info("=" * 60)

    # Download
    tickers = ["SPY", "EFA", "IEF"]
    prices  = download(tickers, period="5y")
    if "SPY" not in prices:
        log.error("SPY download failed. Aborting.")
        return

    spy = prices["SPY"]
    log.info("SPY: %d bars (%s → %s)", len(spy),
             spy.index[0].date(), spy.index[-1].date())

    # ── Strategy 1: IBS ──
    log.info("\n▸ Backtesting SPY IBS Mean Reversion...")
    ibs_sigs = spy_ibs_signals(spy)
    ibs_res  = backtest_ibs(spy, ibs_sigs)

    # SPY buy-and-hold for comparison
    spy_close = spy["Close"].iloc[1:]
    spy_bh_ret = (spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0] * 100
    spy_bh_ret_ann = ((1 + spy_bh_ret/100) ** (1/max(len(spy_close)/252, 0.01)) - 1) * 100

    print("\n" + "=" * 65)
    print("  STRATEGY 1 — SPY IBS Mean Reversion")
    print("=" * 65)
    print(f"  Period:        {ibs_res['start']} → {ibs_res['end']}")
    print(f"  Capital:       ${CAPITAL:,.0f} → ${ibs_res['final_nav']:,.2f}")
    print(f"  Total Return:  {ibs_res['total_return']:+.2f}%  (SPY B&H: {spy_bh_ret:+.1f}%)")
    print(f"  Annual Return: {ibs_res['annual_return']:+.2f}%  (SPY: {spy_bh_ret_ann:+.1f}%)")
    print(f"  Sharpe:        {ibs_res['sharpe']:.3f}")
    print(f"  Max Drawdown:  {ibs_res['max_drawdown']:.2f}%")
    print(f"  Volatility:    {ibs_res['volatility']:.2f}%")
    print(f"  Total Trades:  {ibs_res['total_trades']}")
    print(f"  Win Rate:      {ibs_res['win_rate']:.1f}%")
    print(f"  Profit Factor: {ibs_res['profit_factor']:.2f}")
    print(f"  Avg Win:       ${ibs_res['avg_win']:+.2f}")
    print(f"  Avg Loss:      ${ibs_res['avg_loss']:+.2f}")

    # What's the CURRENT signal?
    last_ibs = float((spy["Close"].iloc[-1] - spy["Low"].iloc[-1]) /
                     max(spy["High"].iloc[-1] - spy["Low"].iloc[-1], 0.001))
    thr_val  = float(spy["High"].rolling(10).max().iloc[-1] -
                     2.5 * (spy["High"] - spy["Low"]).rolling(25).mean().iloc[-1])
    current_sig = ibs_sigs.iloc[-1]
    print(f"\n  Current IBS:   {last_ibs:.3f}  (threshold: <{0.30})")
    print(f"  SPY Close:     ${spy['Close'].iloc[-1]:.2f}  (entry threshold: ${thr_val:.2f})")
    print(f"  TODAY'S SIGNAL: {'LONG ▲' if current_sig == 1 else 'FLAT —'}")

    # ── Strategy 2: GEM ──
    log.info("\n▸ Backtesting Dual Momentum GEM...")
    gem_df  = gem_signals(prices)
    gem_res = backtest_gem(prices, gem_df)

    current_gem = gem_df["hold"].iloc[-1] if len(gem_df) > 0 else "?"
    print("\n" + "=" * 65)
    print("  STRATEGY 2 — Dual Momentum GEM")
    print("=" * 65)
    print(f"  Final NAV:     ${gem_res['final_nav']:,.2f}")
    print(f"  Total Return:  {gem_res['total_return']:+.2f}%")
    print(f"  Annual Return: {gem_res['annual_return']:+.2f}%")
    print(f"  Sharpe:        {gem_res['sharpe']:.3f}")
    print(f"  Max Drawdown:  {gem_res['max_drawdown']:.2f}%")
    print(f"  Trades:        {gem_res['trades']} (monthly rebalances)")
    print(f"\n  CURRENT ALLOCATION: {current_gem}")
    if len(gem_df) > 0:
        last = gem_df.iloc[-1]
        print(f"  SPY 12m return: {last['spy_ret_12m']:.1f}%  |  EFA: {last['efa_ret_12m']:.1f}%")

    # ── Combined verdict ──
    ibs_pass = ibs_res["sharpe"] >= 1.0
    print("\n" + "=" * 65)
    print(f"  IBS Sharpe {ibs_res['sharpe']:.2f}: {'PASS ✓' if ibs_pass else 'NEEDS REVIEW'}")
    print(f"  GEM Sharpe {gem_res['sharpe']:.2f}")
    print("=" * 65)

    # ── Chart ──
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                                  gridspec_kw={"height_ratios": [3, 2, 1]})

        # Top: both strategies vs SPY
        ax1 = axes[0]
        ibs_eq = ibs_res["_equity"]
        gem_eq = gem_res["_equity"]
        spy_eq = ibs_res["_spy_bh"]
        common = ibs_eq.index.intersection(gem_eq.index).intersection(spy_eq.index)
        ax1.plot(common, ibs_eq.loc[common], label=f"IBS Reversion (Sharpe {ibs_res['sharpe']:.2f})",
                 lw=1.5, color="#2196F3")
        ax1.plot(common, gem_eq.loc[common], label=f"Dual Momentum GEM (Sharpe {gem_res['sharpe']:.2f})",
                 lw=1.5, color="#4CAF50")
        ax1.plot(common, spy_eq.loc[common], label="SPY Buy & Hold",
                 lw=1.0, color="#9E9E9E", alpha=0.7)
        ax1.axhline(CAPITAL, color="#999", ls="--", alpha=0.4, lw=0.8)
        ax1.set_title("ATNN Evidence-Based Strategy Backtest", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio ($)"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        # Middle: drawdowns
        ax2 = axes[1]
        for eq, color, label in [(ibs_eq, "#2196F3", "IBS"), (gem_eq, "#4CAF50", "GEM"),
                                   (spy_eq, "#9E9E9E", "SPY")]:
            dd = (eq - eq.cummax()) / eq.cummax()
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.2, color=color)
            ax2.plot(dd.index, dd.values, color=color, lw=0.8, label=label)
        ax2.set_title("Drawdowns"); ax2.set_ylabel("DD %"); ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Bottom: GEM allocation
        ax3 = axes[2]
        if len(gem_df) > 0:
            allocation_map = {"SPY": 1.0, "EFA": 0.5, "IEF": 0.0}
            gem_alloc = gem_df["hold"].map(allocation_map)
            ax3.step(gem_alloc.index, gem_alloc.values, where="post",
                     color="#4CAF50", lw=1.5)
            ax3.set_yticks([0, 0.5, 1.0])
            ax3.set_yticklabels(["IEF\n(Bonds)", "EFA\n(Intl)", "SPY\n(US)"])
        ax3.set_title("GEM Allocation"); ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        path = str(OUTPUT / "equity_curve_final.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Chart → %s", path)
    except Exception as e:
        log.error("Chart: %s", e)

    # ── Save results ──
    result = {
        "timestamp": datetime.now().isoformat(),
        "ibs": {k: v for k, v in ibs_res.items() if not k.startswith("_")},
        "gem": {k: v for k, v in gem_res.items() if not k.startswith("_")},
        "current_signals": {
            "ibs": "LONG" if current_sig == 1 else "FLAT",
            "gem": current_gem,
            "ibs_value": round(last_ibs, 4),
            "spy_close": round(float(spy["Close"].iloc[-1]), 2),
            "entry_threshold": round(thr_val, 2),
        },
        "deploy": ibs_pass,
    }
    with open(str(OUTPUT / "backtest_final.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("Results → backtest_final.json")
    return result


if __name__ == "__main__":
    main()
