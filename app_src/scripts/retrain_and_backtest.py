#!/usr/bin/env python3
"""
retrain_and_backtest.py — OPTIMIZED v2
=======================================
End-to-end walk-forward backtest for ATNN v2.

Key fix: sequence builder uses full historical data up to each date,
not just data within the fold window. This ensures test periods
(even short 21-day windows) can produce valid sequences.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
APP_SRC = SCRIPT_DIR.parent
sys.path.insert(0, str(APP_SRC))

import torch
import yfinance as yf

from nn.features import NNFeatureEngine
from nn.data_loader import direction_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain")

# ─── Configuration ───────────────────────────────────────────────────────

SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "AMD", "JPM", "BAC", "XOM", "GLD", "IWM", "NFLX", "CRM", "AVGO",
    "INTC", "XLF",
]

INITIAL_CAPITAL = 6003.0
MAX_POSITION_PCT = 0.05
TRAIN_WINDOW = 504       # ~2 years of trading days
PREDICT_HORIZON = 21
PURGE_GAP = 5
EMBARGO = 5
SEQ_LEN = 30
HIDDEN = 32
LAYERS = 1
DROPOUT = 0.3
EPOCHS = 15
PATIENCE = 3
LR = 1e-3
BATCH = 256
DIR_THRESH = 0.005

DATA_START = "2022-04-01"
DATA_END = "2026-04-01"

OUTPUT = Path("/home/user/workspace")
MODELS = Path("/home/user/workspace/atnn-bot/models")
MODELS.mkdir(parents=True, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────

def download_data(symbols, start, end):
    logger.info("Downloading %d symbols: %s → %s", len(symbols), start, end)
    data = yf.download(symbols, start=start, end=end,
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                df = data[sym].dropna(how="all")
                if len(df) > 200:
                    result[sym] = df
            except KeyError:
                pass
    else:
        result[symbols[0]] = data.dropna(how="all")
    logger.info("Got %d/%d symbols", len(result), len(symbols))
    return result


def build_features(price_data):
    """Build per-symbol feature DataFrames + targets. 
    Returns: {sym: (features_df, targets_series)}
    """
    engine = NNFeatureEngine(return_windows=[1, 5, 10, 21], vol_windows=[5, 10, 21])
    sym_data = {}
    
    for sym, df in price_data.items():
        try:
            cc = "Close" if "Close" in df.columns else "close"
            vc = "Volume" if "Volume" in df.columns else "volume"
            close = df[cc].dropna()
            vol = df[vc].dropna() if vc in df.columns else None
            
            if len(close) < TRAIN_WINDOW + 50:
                continue
            
            vol_df = pd.DataFrame({vc: vol}) if vol is not None and len(vol) > 0 else None
            feat = engine.build_features(pd.DataFrame({cc: close}), vol_df)
            
            returns = close.pct_change(1).shift(-1)
            labels = direction_labels(returns, threshold=DIR_THRESH)
            
            common = feat.index.intersection(labels.dropna().index)
            feat = feat.loc[common]
            labels = labels.loc[common]
            
            if len(feat) < TRAIN_WINDOW:
                continue
            
            sym_data[sym] = (feat, labels)
            logger.info("  %s: %d rows, %d features", sym, len(feat), feat.shape[1])
        except Exception as e:
            logger.error("  %s: %s", sym, e)
    
    return sym_data


def make_sequences_for_dates(feat_df, labels, target_dates, seq_len):
    """Build sequences for specific target dates using all available history.
    
    For each target_date in target_dates that exists in feat_df.index,
    we take the seq_len rows ending at (and including) the row BEFORE target_date
    and use the target at target_date as the label.
    
    This way even a 1-day test window produces valid sequences as long as
    there's enough history.
    """
    all_dates = feat_df.index.sort_values()
    values = feat_df.values.astype(np.float64)
    tgts = labels.values.astype(np.int64)
    
    # Expanding z-score on full history
    cumsum = np.cumsum(values, axis=0)
    cumsum2 = np.cumsum(values ** 2, axis=0)
    counts = np.arange(1, len(values) + 1).reshape(-1, 1)
    means = cumsum / counts
    variances = cumsum2 / counts - means ** 2
    stds = np.sqrt(np.maximum(variances, 0))
    stds[stds == 0] = 1.0
    normalized = np.nan_to_num((values - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    
    X, y, dates_out = [], [], []
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    for d in target_dates:
        if d not in date_to_idx:
            continue
        idx = date_to_idx[d]
        if idx < seq_len:
            continue
        seq = normalized[idx - seq_len:idx]
        X.append(seq)
        y.append(tgts[idx])
        dates_out.append(d)
    
    if not X:
        return np.array([]), np.array([]), []
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), dates_out


def train_model(model, X_tr, y_tr, X_val, y_val, device):
    """Train with early stopping, return val_loss."""
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    
    Xt = torch.tensor(X_tr, device=device)
    yt = torch.tensor(y_tr, device=device)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, device=device)
    n = len(Xt)
    
    best = float("inf")
    wait = 0
    
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xt[idx], yt[idx]
            lens = torch.full((len(idx),), SEQ_LEN, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = crit(model(xb, lengths=lens), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        
        model.eval()
        vl = []
        with torch.no_grad():
            for i in range(0, len(Xv), BATCH):
                xb = Xv[i:i+BATCH]
                yb_v = yv[i:i+BATCH]
                lens = torch.full((len(xb),), SEQ_LEN, dtype=torch.long, device=device)
                vl.append(crit(model(xb, lengths=lens), yb_v).item())
        
        val_loss = float(np.mean(vl)) if vl else float("inf")
        sched.step(val_loss)
        
        if val_loss < best:
            best = val_loss
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    
    return best


def get_predictions(model, X, device):
    model.eval()
    preds, confs = [], []
    Xt = torch.tensor(X, device=device)
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            xb = Xt[i:i+BATCH]
            lens = torch.full((len(xb),), SEQ_LEN, dtype=torch.long, device=device)
            probs = torch.softmax(model(xb, lengths=lens), dim=-1)
            preds.append(probs.argmax(1).cpu().numpy())
            confs.append(probs.max(1).values.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


# ─── Main pipeline ───────────────────────────────────────────────────────

def main():
    t0 = time.time()
    device = torch.device("cpu")
    
    from nn.models.lstm_predictor import LSTMPredictor
    
    # 1. Download
    logger.info("═══ STEP 1: Download ═══")
    price_data = download_data(SYMBOLS, DATA_START, DATA_END)
    
    # 2. Features
    logger.info("═══ STEP 2: Features ═══")
    sym_data = build_features(price_data)
    n_features = next(iter(sym_data.values()))[0].shape[1]
    logger.info("Symbols: %d, Features: %d", len(sym_data), n_features)
    
    # Get all unique dates across symbols
    all_dates_set = set()
    for feat, _ in sym_data.values():
        all_dates_set.update(feat.index)
    unique_dates = sorted(all_dates_set)
    n_dates = len(unique_dates)
    logger.info("Date range: %s → %s (%d days)", 
                unique_dates[0].date(), unique_dates[-1].date(), n_dates)
    
    # 3. Walk-forward
    logger.info("═══ STEP 3: Walk-Forward Training ═══")
    
    oos_all = {}  # (date, sym) → (pred, conf)
    fold_metrics = []
    best_val = float("inf")
    best_path = ""
    
    fold = 0
    cursor = TRAIN_WINDOW
    
    while cursor + PURGE_GAP + PREDICT_HORIZON <= n_dates:
        train_dates_list = unique_dates[max(0, cursor - TRAIN_WINDOW):cursor]
        test_start = cursor + PURGE_GAP
        test_end = min(test_start + PREDICT_HORIZON, n_dates)
        test_dates_list = unique_dates[test_start:test_end]
        
        if not test_dates_list:
            break
        
        train_set = set(train_dates_list)
        test_set = set(test_dates_list)
        
        # Build pooled train/test sequences across all symbols
        all_X_tr, all_y_tr = [], []
        all_X_te, all_y_te, all_keys_te = [], [], []
        
        for sym, (feat, labels) in sym_data.items():
            # For training: build sequences for dates in train_set
            sym_train_dates = sorted(d for d in train_dates_list if d in feat.index)
            if len(sym_train_dates) < SEQ_LEN + 1:
                continue
            
            X_tr, y_tr, _ = make_sequences_for_dates(feat, labels, sym_train_dates, SEQ_LEN)
            if len(X_tr) > 0:
                all_X_tr.append(X_tr)
                all_y_tr.append(y_tr)
            
            # For testing: build sequences for dates in test_set
            # These use history up to each test date (which includes train period)
            sym_test_dates = sorted(d for d in test_dates_list if d in feat.index)
            X_te, y_te, dates_te = make_sequences_for_dates(feat, labels, sym_test_dates, SEQ_LEN)
            if len(X_te) > 0:
                all_X_te.append(X_te)
                all_y_te.append(y_te)
                all_keys_te.extend([(d, sym) for d in dates_te])
        
        if not all_X_tr or not all_X_te:
            logger.warning("Fold %d: empty train or test, skip", fold)
            cursor += PREDICT_HORIZON + EMBARGO
            fold += 1
            continue
        
        X_tr = np.concatenate(all_X_tr)
        y_tr = np.concatenate(all_y_tr)
        X_te = np.concatenate(all_X_te)
        y_te = np.concatenate(all_y_te)
        
        logger.info("Fold %d: %s→%s | train=%d test=%d seqs",
                     fold, test_dates_list[0].date(), test_dates_list[-1].date(),
                     len(X_tr), len(X_te))
        
        # Train
        model = LSTMPredictor(
            input_size=n_features, hidden_size=HIDDEN,
            num_layers=LAYERS, dropout=DROPOUT, num_classes=3,
        ).to(device)
        
        val_loss = train_model(model, X_tr, y_tr, X_te, y_te, device)
        
        # Predict
        preds, confs = get_predictions(model, X_te, device)
        acc = float((preds == y_te).mean())
        
        for i, key in enumerate(all_keys_te):
            oos_all[key] = (int(preds[i]), float(confs[i]))
        
        fold_metrics.append({
            "fold": fold,
            "test_start": str(test_dates_list[0].date()),
            "test_end": str(test_dates_list[-1].date()),
            "train_seqs": len(X_tr),
            "test_seqs": len(X_te),
            "val_loss": round(val_loss, 4),
            "accuracy": round(acc, 4),
        })
        logger.info("  acc=%.4f val_loss=%.4f", acc, val_loss)
        
        ckpt = str(MODELS / f"fold{fold}.pt")
        torch.save(model.state_dict(), ckpt)
        if val_loss < best_val:
            best_val = val_loss
            best_path = ckpt
            torch.save({
                "state_dict": model.state_dict(),
                "n_features": n_features,
                "hidden": HIDDEN, "layers": LAYERS,
                "dropout": DROPOUT, "fold": fold,
                "val_loss": val_loss, "accuracy": acc,
            }, str(MODELS / "best_model.pt"))
        
        cursor += PREDICT_HORIZON + EMBARGO
        fold += 1
    
    logger.info("Done: %d folds, %d OOS predictions", fold, len(oos_all))
    
    if not oos_all:
        logger.error("No OOS predictions generated!")
        return
    
    # 4. Backtest
    logger.info("═══ STEP 4: Backtest ═══")
    
    pred_dates = sorted(set(d for (d, s) in oos_all))
    pred_syms = sorted(set(s for (d, s) in oos_all))
    
    signals = pd.DataFrame(0.0, index=pred_dates, columns=pred_syms)
    for (d, s), (pred, conf) in oos_all.items():
        if pred == 2:     # UP → long
            signals.loc[d, s] = conf
        elif pred == 0:   # DOWN → short
            signals.loc[d, s] = -conf
    
    # Price bars
    bars = {}
    for sym, df in price_data.items():
        if sym not in pred_syms:
            continue
        oc = "Open" if "Open" in df.columns else "open"
        hc = "High" if "High" in df.columns else "high"
        lc = "Low" if "Low" in df.columns else "low"
        cc = "Close" if "Close" in df.columns else "close"
        for date in df.index:
            if date not in bars:
                bars[date] = {}
            try:
                bars[date][sym] = {
                    "open": float(df.loc[date, oc]),
                    "high": float(df.loc[date, hc]),
                    "low": float(df.loc[date, lc]),
                    "close": float(df.loc[date, cc]),
                }
            except:
                pass
    
    # Position sizer
    def sizer(sig, nav, sym):
        alloc = nav * MAX_POSITION_PCT * min(abs(sig), 1.0)
        # Get latest price
        for d in sorted(bars.keys(), reverse=True):
            if sym in bars[d]:
                p = bars[d][sym]["close"]
                return max(int(alloc / p), 0) if p > 0 else 0
        return 0
    
    from backtest.engine import BacktestEngine
    engine = BacktestEngine(equity_slippage=0.001, respect_market_hours=False)
    result = engine.run(signals=signals, price_data=bars,
                        initial_capital=INITIAL_CAPITAL, position_sizer=sizer)
    
    # 5. SPY benchmark
    spy_eq = pd.Series(dtype=float)
    if "SPY" in price_data:
        cc = "Close" if "Close" in price_data["SPY"].columns else "close"
        sp = price_data["SPY"][cc].loc[pred_dates[0]:pred_dates[-1]]
        if len(sp) > 1:
            spy_eq = sp * (INITIAL_CAPITAL / sp.iloc[0])
    
    # 6. Report
    logger.info("═══ STEP 5: Report ═══")
    m = result.metrics
    eq = result.equity_curve
    
    print("\n" + "=" * 70)
    print("  ATNN v2 — WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)
    
    if len(eq) > 0:
        print(f"\n{'Initial Capital:':<30} ${INITIAL_CAPITAL:,.2f}")
        print(f"{'Final NAV:':<30} ${eq.iloc[-1]:,.2f}")
        print(f"{'Period:':<30} {eq.index[0].date()} → {eq.index[-1].date()}")
    
    print(f"\n{'Total Return:':<30} {m.get('total_return',0)*100:+.2f}%")
    print(f"{'Annual Return:':<30} {m.get('annual_return',0)*100:+.2f}%")
    print(f"{'Sharpe Ratio:':<30} {m.get('sharpe_ratio',0):.4f}")
    print(f"{'Sortino Ratio:':<30} {m.get('sortino_ratio',0):.4f}")
    print(f"{'Max Drawdown:':<30} {m.get('max_drawdown',0)*100:.2f}%")
    print(f"{'Volatility:':<30} {m.get('volatility',0)*100:.2f}%")
    print(f"{'Total Trades:':<30} {m.get('total_trades',0)}")
    print(f"{'Win Rate:':<30} {m.get('win_rate',0)*100:.1f}%")
    print(f"{'Profit Factor:':<30} {m.get('profit_factor',0):.2f}")
    print(f"{'Avg Holding:':<30} {m.get('avg_holding_period',0):.1f} days")
    
    if len(spy_eq) > 1:
        spy_ret = (spy_eq.iloc[-1] - spy_eq.iloc[0]) / spy_eq.iloc[0]
        print(f"\n{'SPY Buy&Hold:':<30} {spy_ret*100:+.2f}%")
        print(f"{'Alpha:':<30} {(m.get('total_return',0) - spy_ret)*100:+.2f}%")
    
    print(f"\n{'— Fold Metrics —':^70}")
    accs = []
    for fm in fold_metrics:
        print(f"  Fold {fm['fold']:>2}: {fm['test_start']} → {fm['test_end']} | "
              f"acc={fm['accuracy']:.4f} | loss={fm['val_loss']:.4f} | "
              f"train={fm['train_seqs']:>5} test={fm['test_seqs']:>4}")
        accs.append(fm['accuracy'])
    
    if accs:
        print(f"\n  Avg Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    
    sharpe = m.get("sharpe_ratio", 0)
    print(f"\n{'='*70}")
    if sharpe > 1.0:
        print(f"  PASS — Sharpe {sharpe:.2f} > 1.0. Ready for deployment.")
    elif sharpe > 0.5:
        print(f"  MARGINAL — Sharpe {sharpe:.2f}. Needs tuning.")
    else:
        print(f"  NEEDS WORK — Sharpe {sharpe:.2f}")
        if accs and np.mean(accs) < 0.38:
            print("  → Accuracy near random. Model needs better features.")
        if m.get("total_trades", 0) < 20:
            print("  → Very few trades. Consider lowering signal threshold.")
        pf = m.get("profit_factor", 0)
        if 0 < pf < 1:
            print(f"  → Profit factor {pf:.2f} < 1. Losses exceed wins.")
    print("=" * 70)
    
    # Save chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={"height_ratios": [3, 1]})
        if len(eq) > 0:
            ax1.plot(eq.index, eq.values, label="ATNN v2", lw=1.5, color="#2196F3")
        if len(spy_eq) > 1:
            common = eq.index.intersection(spy_eq.index)
            if len(common) > 0:
                ax1.plot(common, spy_eq.loc[common], label="SPY B&H",
                         lw=1, color="#757575", alpha=.7)
        ax1.axhline(INITIAL_CAPITAL, color="#999", ls="--", alpha=.5)
        ax1.set_title("ATNN v2 Walk-Forward Backtest", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio ($)")
        ax1.legend()
        ax1.grid(True, alpha=.3)
        
        if len(eq) > 1:
            dd = (eq - eq.cummax()) / eq.cummax()
            ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=.3)
            ax2.plot(dd.index, dd.values, color="#F44336", lw=.8)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("DD %")
        ax2.grid(True, alpha=.3)
        
        plt.tight_layout()
        plt.savefig(str(OUTPUT / "equity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Chart → equity_curve.png")
    except Exception as e:
        logger.error("Chart: %s", e)
    
    # Save CSV + JSON
    pd.DataFrame(fold_metrics).to_csv(str(OUTPUT / "fold_metrics.csv"), index=False)
    
    rj = {
        "timestamp": datetime.now().isoformat(),
        "initial_capital": INITIAL_CAPITAL,
        "final_nav": float(eq.iloc[-1]) if len(eq) > 0 else 0,
        "n_symbols": len(pred_syms),
        "n_folds": len(fold_metrics),
        "n_oos": len(oos_all),
        "avg_accuracy": float(np.mean(accs)) if accs else 0,
        "metrics": {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
            for k, v in m.items()
            if not isinstance(v, (pd.DataFrame, pd.Series, list))
        },
        "verdict": "PASS" if sharpe > 1.0 else ("MARGINAL" if sharpe > .5 else "NEEDS_WORK"),
        "best_model": best_path,
    }
    with open(str(OUTPUT / "backtest_results.json"), "w") as f:
        json.dump(rj, f, indent=2, default=str)
    
    logger.info("Done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
