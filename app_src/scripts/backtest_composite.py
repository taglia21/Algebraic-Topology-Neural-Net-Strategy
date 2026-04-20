#!/usr/bin/env python3
"""
backtest_composite.py — Compare composite TDA scorer vs single-feature baseline.

Runs both systems on 4 years of SPY daily data with full MES costs.
Reports: Sharpe, IC, win rate, profit factor for each.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from tda.extractor import TDAFeatureExtractor
from tda.composite_scorer import TDACompositeScorer
from nn.regime_labeler import heuristic_regime, HEURISTIC_THRESHOLDS

CAPITAL = 5923.0
COST_PER_TRADE = 4.25  # commission + spread + slippage per MES contract
MES_MULT = 5.0
ATR_STOP_MULT = 2.0

def run():
    print("Downloading SPY (4 years)...")
    spy = yf.download("SPY", period="4y", interval="1d", auto_adjust=True, progress=False)
    close = spy["Close"].squeeze()
    high  = spy["High"].squeeze()
    low   = spy["Low"].squeeze()

    print("Computing TDA features...")
    ext = TDAFeatureExtractor(window=40, stride=1)
    tda = ext.extract_series(close)
    print(f"TDA: {len(tda)} rows, {tda.shape[1]} features")

    # ── Composite scorer ──
    print("\nFitting composite scorer (walk-forward)...")
    scorer = TDACompositeScorer(train_window=120, forward_bars=5, retrain_every=20)
    scores = scorer.fit_and_score(tda, close)

    # ── Baseline: single spectral-gap threshold ──
    log_ret = np.log(close / close.shift(1))
    mom_5   = close.pct_change(5)
    vol_10  = log_ret.rolling(10).std()
    atr_14  = (close - close.shift(1)).abs().rolling(14).mean()

    baseline_signal = pd.Series(0.0, index=tda.index)
    for i, date in enumerate(tda.index):
        if date not in close.index:
            continue
        idx = close.index.get_loc(date)
        sg = float(tda.loc[date, "spectral_gap"])
        b1 = float(tda.loc[date, "beta_1"])
        wd = float(tda.loc[date, "wasserstein_dist"])
        m5 = float(mom_5.iloc[idx]) if not pd.isna(mom_5.iloc[idx]) else 0
        vol = float(vol_10.iloc[idx]) if not pd.isna(vol_10.iloc[idx]) else 0.015
        regime, conf = heuristic_regime(sg, b1, wd, m5, vol)
        if regime == 0:  # TRENDING_UP
            baseline_signal.iloc[i] = 1.0

    # ── Simulate both ──
    def simulate(signal_series, name, entry_threshold=0.5):
        nav = CAPITAL
        pos = 0
        entry_px = 0.0
        stop_px = 0.0
        trades = []
        equity = []

        for i, date in enumerate(tda.index):
            if date not in close.index:
                equity.append(nav)
                continue

            idx = close.index.get_loc(date)
            px = float(close.iloc[idx])
            sig = float(signal_series.iloc[i])

            # Check stop
            if pos > 0 and px <= stop_px:
                pnl = (stop_px - entry_px) * 10 * pos * MES_MULT - COST_PER_TRADE * pos
                trades.append(pnl)
                nav += pnl
                pos = 0

            # Unrealized
            unrealized = pos * (px - entry_px) * 10 * MES_MULT if pos > 0 else 0
            equity.append(nav + unrealized)

            if i < 60:
                continue

            atr = float(atr_14.iloc[idx]) if not pd.isna(atr_14.iloc[idx]) else 5.0

            # Entry
            if pos == 0 and sig > entry_threshold:
                qty = 2 if sig > 0.75 else 1
                entry_px = px
                stop_px = px - ATR_STOP_MULT * atr
                pos = qty
                nav -= COST_PER_TRADE * qty

            # Exit on signal drop
            elif pos > 0 and sig < entry_threshold * 0.8:
                pnl = (px - entry_px) * 10 * pos * MES_MULT - COST_PER_TRADE * pos
                trades.append(pnl)
                nav += pnl
                pos = 0

        # Close final
        if pos > 0:
            px = float(close.iloc[-1])
            pnl = (px - entry_px) * 10 * pos * MES_MULT - COST_PER_TRADE * pos
            trades.append(pnl)
            nav += pnl

        eq = pd.Series(equity, index=tda.index[:len(equity)])
        ret = eq.pct_change().dropna()
        total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        years = len(ret) / 252
        ann = (1 + total) ** (1/max(years, 0.01)) - 1
        vol = float(ret.std() * np.sqrt(252))
        sharpe = float((ret.mean() - 0.05/252) / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
        dd = (eq - eq.cummax()) / eq.cummax()

        pnls = np.array(trades)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        # IC: correlation of signal with 5-day forward return
        fwd5 = close.pct_change(5).shift(-5).reindex(tda.index)
        common = signal_series.dropna().index.intersection(fwd5.dropna().index)
        if len(common) > 20:
            ic, p_ic = stats.spearmanr(signal_series.loc[common], fwd5.loc[common])
        else:
            ic, p_ic = 0, 1

        return {
            "name": name,
            "total_return": round(total * 100, 2),
            "annual_return": round(ann * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(float(dd.min()) * 100, 2),
            "trades": len(trades),
            "win_rate": round(len(wins)/max(len(trades),1)*100, 1),
            "profit_factor": round(abs(wins.sum()/losses.sum()) if len(losses) and losses.sum() != 0 else 0, 2),
            "avg_win": round(float(wins.mean()) if len(wins) else 0, 2),
            "avg_loss": round(float(losses.mean()) if len(losses) else 0, 2),
            "ic": round(float(ic), 4),
            "ic_pval": round(float(p_ic), 4),
            "_equity": eq,
        }

    print("\nSimulating baseline (single spectral-gap)...")
    base = simulate(baseline_signal, "Baseline (spectral_gap)", entry_threshold=0.5)

    print("Simulating composite scorer...")
    comp = simulate(scores, "Composite (5-feature)", entry_threshold=0.60)

    # SPY buy-and-hold
    spy_bh = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

    print("\n" + "=" * 70)
    print("  COMPOSITE vs BASELINE — HEAD-TO-HEAD")
    print("=" * 70)

    header = f"{'Metric':<25} {'Baseline':>15} {'Composite':>15} {'Improvement':>15}"
    print(header)
    print("-" * 70)

    metrics = ["total_return", "annual_return", "sharpe", "max_drawdown",
               "trades", "win_rate", "profit_factor", "ic"]
    labels  = ["Total Return %", "Annual Return %", "Sharpe", "Max Drawdown %",
               "Trades", "Win Rate %", "Profit Factor", "Information Coeff"]

    for metric, label in zip(metrics, labels):
        b = base[metric]
        c = comp[metric]
        if isinstance(b, (int, float)) and isinstance(c, (int, float)) and b != 0:
            imp = f"{(c-b)/abs(b)*100:+.0f}%" if b != 0 else "N/A"
        else:
            imp = ""
        print(f"  {label:<23} {str(b):>15} {str(c):>15} {imp:>15}")

    print(f"\n  SPY Buy & Hold: {spy_bh:+.1f}%")
    print(f"  Composite IC p-value: {comp['ic_pval']}")

    verdict = "PASS" if comp["sharpe"] > 1.2 else "CLOSE" if comp["sharpe"] > 0.9 else "NEEDS WORK"
    print(f"\n  Verdict: {verdict} (target Sharpe > 1.2)")
    print("=" * 70)

    # Chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 10),
                                      gridspec_kw={"height_ratios": [3, 1]})
        spy_eq = close.reindex(comp["_equity"].index)
        spy_eq = spy_eq * (CAPITAL / spy_eq.iloc[0])
        a1.plot(comp["_equity"].index, comp["_equity"].values,
                label=f"Composite (Sharpe {comp['sharpe']})", lw=1.5, color="#2196F3")
        a1.plot(base["_equity"].index, base["_equity"].values,
                label=f"Baseline (Sharpe {base['sharpe']})", lw=1.2, color="#FF9800")
        a1.plot(spy_eq.index, spy_eq.values, label="SPY B&H", lw=0.8, color="#9E9E9E", alpha=0.7)
        a1.axhline(CAPITAL, color="#999", ls="--", alpha=0.4)
        a1.set_title("Composite TDA Scorer vs Baseline", fontsize=14, fontweight="bold")
        a1.set_ylabel("$"); a1.legend(); a1.grid(True, alpha=0.3)

        a2.plot(scores.index, scores.values, lw=0.8, color="#2196F3", alpha=0.7)
        a2.axhline(0.60, color="green", ls="--", alpha=0.5, label="Entry threshold")
        a2.axhline(0.50, color="gray", ls=":", alpha=0.5)
        a2.set_title("Composite Score Over Time"); a2.set_ylabel("Score")
        a2.legend(); a2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("/home/user/workspace/composite_vs_baseline.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\n  Chart saved: composite_vs_baseline.png")
    except Exception as e:
        print(f"\n  Chart error: {e}")

if __name__ == "__main__":
    run()
