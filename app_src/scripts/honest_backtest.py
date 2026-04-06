#!/usr/bin/env python3
"""
honest_backtest.py
==================
Walk-forward P&L backtest — no look-ahead bias, full costs, correct stops.
Uses the heuristic regime (worst-case: no trained TCN) + IBS overlay.
All prices in consistent units (SPY, not SPX mix).
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, yfinance as yf
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nn.regime_labeler import heuristic_regime, regime_to_contracts
from tda.extractor import TDAFeatureExtractor

# ── Config ────────────────────────────────────────────────────────
CAPITAL      = 5923.0
COMM_RT      = 0.50   # IBKR commission per contract
SPREAD_RT    = 2.50   # bid-ask spread
SLIP_RT      = 1.25   # slippage
TOTAL_RT     = COMM_RT + SPREAD_RT + SLIP_RT   # $4.25 per contract round-trip
MES_MULT     = 5.0                              # $5/pt for 1 MES
ATR_STOP_MULT = 2.5
MIN_CONF     = 0.55
MAX_CONTRACTS = 2

IBS_ATR_WIN, IBS_HI_WIN = 25, 10
IBS_MULT, IBS_THRESH     = 2.5, 0.30
SMOOTH_WINDOW            = 3


def run():
    print("Downloading 4-year SPY data...")
    spy  = yf.download("SPY", period="4y", interval="1d", auto_adjust=True, progress=False)
    close = spy["Close"].squeeze()
    high  = spy["High"].squeeze()
    low   = spy["Low"].squeeze()

    log_ret = np.log(close / close.shift(1))
    mom_5   = close.pct_change(5)
    vol_10  = log_ret.rolling(10).std()
    atr_14  = (close - close.shift(1)).abs().rolling(14).mean()  # SPY points

    # IBS signal (SPY price-based, consistent)
    ibs       = (close - low) / (high - low).replace(0, np.nan)
    avg_rng   = (high - low).rolling(IBS_ATR_WIN).mean()
    roll_hi   = high.rolling(IBS_HI_WIN).max()
    ibs_thr   = roll_hi - IBS_MULT * avg_rng          # SPY price
    ibs_entry = (close < ibs_thr) & (ibs < IBS_THRESH)

    # TDA features
    ext = TDAFeatureExtractor(window=40, stride=1)
    tda = ext.extract_series(close)

    # ── Walk-forward loop ──────────────────────────────────────────
    nav       = CAPITAL
    pos       = 0              # contracts held (SPY-denominated)
    entry_px  = 0.0            # SPY entry price
    entry_i   = 0
    stop_px   = 0.0            # SPY stop price
    ibs_active = False
    reg_history = []
    trades    = []
    equity    = []
    warmup    = 54

    for i, date in enumerate(close.index):
        px  = float(close.iloc[i])   # SPY price — ALL comparisons in SPY

        # ── 1. Check stop-loss (SPY units) ──
        if pos > 0 and i > 0:
            if px <= stop_px:
                # Stopped out: P&L = (stop_px - entry_px) × qty × MES_MULT × 10
                # SPY → SPX conversion: ×10 → SPX points × $5/pt
                pnl = (stop_px - entry_px) * 10 * pos * MES_MULT - TOTAL_RT * pos
                trades.append({"pnl": pnl, "type": "STOP",
                                "hold": i - entry_i, "entry": entry_px, "exit": stop_px})
                nav += pnl
                pos = 0
                ibs_active = False

        # ── 2. IBS exit: close > previous bar's HIGH ──
        if ibs_active and pos > 0 and i > 0:
            if px > float(high.iloc[i - 1]):
                pnl = (px - entry_px) * 10 * pos * MES_MULT - TOTAL_RT * pos
                trades.append({"pnl": pnl, "type": "IBS_EXIT",
                                "hold": i - entry_i, "entry": entry_px, "exit": px})
                nav += pnl
                pos = 0
                ibs_active = False

        # Record current equity (unrealized P&L only — NOT full notional)
        # Futures: we hold margin, not notional. Equity = cash + open trade P&L.
        unrealized = pos * (px - entry_px) * 10 * MES_MULT if (pos > 0 and entry_px > 0) else 0.0
        equity.append(nav + unrealized)

        # ── 3. Signal computation (no look-ahead) ──
        if i < warmup or date not in tda.index:
            continue

        row   = tda.loc[date]
        sg    = float(row.get("spectral_gap", 0.5))
        b1    = float(row.get("beta_1", 0))
        w     = float(row.get("wasserstein_dist", 0))
        m5    = float(mom_5.iloc[i]) if not pd.isna(mom_5.iloc[i]) else 0
        vol   = float(vol_10.iloc[i]) if not pd.isna(vol_10.iloc[i]) else 0.015
        atr   = float(atr_14.iloc[i]) if not pd.isna(atr_14.iloc[i]) else 7.0

        regime, conf = heuristic_regime(sg, b1, w, m5, vol)

        # 3-bar smoothing
        reg_history.append(regime)
        reg_history = reg_history[-SMOOTH_WINDOW:]
        regime = Counter(reg_history).most_common(1)[0][0]

        # ── 4. IBS entry (priority over regime) ──
        if not ibs_active and pos == 0 and bool(ibs_entry.iloc[i]):
            entry_px  = px                           # SPY price
            entry_i   = i
            stop_px   = entry_px - ATR_STOP_MULT * atr   # SPY price
            pos       = 1                            # always 1 contract for IBS
            ibs_active = True
            nav      -= TOTAL_RT * pos
            continue

        # ── 5. Regime-driven entry/exit ──
        if not ibs_active:
            # Use SPY price (not SPX) for contract sizing
            qty = regime_to_contracts(
                regime, conf, nav,
                mes_price=px * 10,    # SPX for dollar-value calculation
                max_contracts=MAX_CONTRACTS,
                min_confidence=MIN_CONF,
            )

            if qty != pos:
                if pos > 0 and qty == 0:
                    # Exit existing position
                    pnl = (px - entry_px) * 10 * pos * MES_MULT - TOTAL_RT * pos
                    trades.append({"pnl": pnl, "type": "REGIME_EXIT",
                                   "hold": i - entry_i, "entry": entry_px, "exit": px})
                    nav += pnl
                    pos  = 0

                elif qty > 0 and pos == 0:
                    # Enter new position
                    entry_px = px
                    entry_i  = i
                    stop_px  = entry_px - ATR_STOP_MULT * atr  # SPY units
                    pos      = qty
                    nav     -= TOTAL_RT * qty

    # ── Close any remaining position at last price ──
    if pos > 0:
        px  = float(close.iloc[-1])
        pnl = (px - entry_px) * 10 * pos * MES_MULT - TOTAL_RT * pos
        trades.append({"pnl": pnl, "type": "EOD", "hold": len(close)-entry_i,
                       "entry": entry_px, "exit": px})
        nav += pnl

    # ── Compute metrics ───────────────────────────────────────────
    eq   = pd.Series(equity, index=close.index[:len(equity)])
    ret  = eq.pct_change().dropna()
    total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
    years = len(ret) / 252
    ann   = (1 + total) ** (1 / max(years, 0.01)) - 1
    vol_a = float(ret.std() * np.sqrt(252))
    rf    = 0.05 / 252
    sharpe = float((ret.mean() - rf) / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    sortino_d = ret[ret < 0]
    sortino = float((ret.mean()-rf)/sortino_d.std()*np.sqrt(252)) if len(sortino_d) > 0 else 0
    dd   = (eq - eq.cummax()) / eq.cummax()
    spy_bh = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

    pnls  = [t["pnl"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    stops  = [t for t in trades if t["type"] == "STOP"]
    ibs_t  = [t for t in trades if "IBS" in t["type"]]
    reg_t  = [t for t in trades if "REGIME" in t["type"] or t["type"] == "EOD"]

    print("\n" + "═"*65)
    print("  HONEST BACKTEST — Full Costs, 2.5×ATR Stops, 3-Bar Smoothing")
    print("═"*65)
    print(f"  ${CAPITAL:,.0f} → ${eq.iloc[-1]:,.2f}")
    print(f"  {eq.index[0].date()} → {eq.index[-1].date()}")
    print(f"\n  Total Return:  {total*100:+.2f}%  (SPY B&H: {spy_bh*100:+.1f}%)")
    print(f"  Annual Return: {ann*100:+.2f}%")
    print(f"  Sharpe:        {sharpe:.3f}")
    print(f"  Sortino:       {sortino:.3f}")
    print(f"  Max Drawdown:  {dd.min()*100:.2f}%")
    print(f"  Volatility:    {vol_a*100:.2f}%")
    print(f"\n  Total Trades:  {len(trades)}")
    if pnls:
        print(f"  Win Rate:      {len(wins)/len(pnls)*100:.1f}%")
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 0
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Avg Win:       ${np.mean(wins):.2f}")
        print(f"  Avg Loss:      ${np.mean(losses):.2f}")
        holds = [t["hold"] for t in trades]
        print(f"  Avg Hold:      {np.mean(holds):.1f} days")
    print(f"  Stops:         {len(stops)} ({len(stops)/max(len(trades),1)*100:.1f}%)")
    print(f"  IBS trades:    {len(ibs_t)}")
    print(f"  Regime trades: {len(reg_t)}")
    print(f"  Total costs:   ${len(trades)*TOTAL_RT:.2f}")

    if sharpe > 1.0:
        verdict = f"PASS — Sharpe {sharpe:.2f}"
    elif sharpe > 0.5:
        verdict = f"MARGINAL — Sharpe {sharpe:.2f}"
    elif sharpe > 0.0:
        verdict = f"WEAK BUT POSITIVE — Sharpe {sharpe:.2f}"
    else:
        verdict = f"FAILING — Sharpe {sharpe:.2f}"

    print(f"\n  {verdict}")
    print("═"*65)

    # Save chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 10),
                                      gridspec_kw={"height_ratios": [3, 1]})
        spy_eq = close[:len(eq)] * (CAPITAL / close.iloc[0])
        a1.plot(eq.index, eq.values, label=f"ATNN ({verdict})", lw=1.5, color="#2196F3")
        a1.plot(eq.index[:len(spy_eq)], spy_eq.values, label="SPY B&H", lw=1, color="#757575", alpha=.7)
        a1.axhline(CAPITAL, color="#999", ls="--", alpha=.4)
        a1.set_title("ATNN Honest Backtest — Heuristic Regime + IBS", fontsize=14, fontweight="bold")
        a1.set_ylabel("$"); a1.legend(); a1.grid(True, alpha=.3)
        a2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=.3)
        a2.set_ylabel("Drawdown"); a2.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig("/home/user/workspace/equity_curve_honest.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\n  Chart saved: equity_curve_honest.png")
    except Exception as e:
        print(f"\n  Chart error: {e}")

    return sharpe


if __name__ == "__main__":
    run()
