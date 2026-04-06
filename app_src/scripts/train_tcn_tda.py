#!/usr/bin/env python3
"""
train_tcn_tda.py — Walk-forward TCN + TDA training pipeline
=============================================================
Downloads 3 years of daily data for the trading universe,
computes TDA features, labels regimes, trains TCN model with
walk-forward validation, and saves the best checkpoint.

Run: python3 /opt/atnn/scripts/train_tcn_tda.py
"""

from __future__ import annotations
import json, logging, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

APP_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(APP_SRC))

import yfinance as yf

from tda.extractor import TDAFeatureExtractor
from nn.models.tcn_predictor import TCNPredictor
from nn.regime_labeler import label_regimes, compute_class_weights

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "AMD", "NFLX", "JPM", "BAC", "XOM", "CVX", "GLD",
]

TDA_WINDOW    = 40       # rolling window for TDA
SEQ_LEN       = 30       # LSTM/TCN sequence length
TRAIN_WINDOW  = 504      # ~2 years training bars
TEST_WINDOW   = 63       # ~3 months test
PURGE         = 5        # purge gap (no-man's-land between train/test)
EPOCHS        = 50
PATIENCE      = 8
LR            = 1e-3
BATCH         = 64
DROPOUT       = 0.2

MODEL_DIR  = Path("/opt/atnn/models")
RESULT_DIR = Path("/opt/atnn/data/training")

FEAT_COLS = [
    "beta_0", "beta_1", "persistence_entropy", "wasserstein_dist",
    "spectral_gap", "sci",
    "mom_5", "mom_20", "vol_10", "rsi", "log_ret",
]


def download_universe() -> dict[str, pd.DataFrame]:
    log.info("Downloading %d symbols...", len(SYMBOLS))
    raw = yf.download(SYMBOLS, period="4y", interval="1d",
                      group_by="ticker", auto_adjust=True,
                      threads=True, progress=False)
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for s in SYMBOLS:
            try:
                df = raw[s].dropna(how="all")
                if len(df) > 300:
                    out[s] = df
            except KeyError:
                pass
    log.info("Got %d symbols", len(out))
    return out


def build_features_for_symbol(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Build full feature matrix (TDA + price) for one symbol."""
    close = df["Close"].dropna()
    high  = df["High"].dropna()
    low   = df["Low"].dropna()

    if len(close) < TDA_WINDOW + SEQ_LEN + 20:
        return pd.DataFrame()

    # TDA
    ext  = TDAFeatureExtractor(window=TDA_WINDOW, stride=1)
    tda  = ext.extract_series(close)

    if len(tda) < SEQ_LEN + 20:
        return pd.DataFrame()

    # Price features
    log_ret = np.log(close / close.shift(1))
    price_feats = pd.DataFrame({
        "mom_5":  close.pct_change(5),
        "mom_20": close.pct_change(20),
        "vol_10": log_ret.rolling(10).std() * np.sqrt(252),
        "rsi":    _rsi(close, 14) / 100.0,
        "log_ret": log_ret,
    })

    combined = pd.concat([tda, price_feats], axis=1)
    combined = combined.ffill().dropna()
    combined["_symbol"] = sym
    return combined


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_sequences(features: pd.DataFrame, labels: pd.Series) -> tuple:
    """Build fixed-length input sequences for TCN."""
    common = features.index.intersection(labels.index)
    if len(common) < SEQ_LEN + 1:
        return np.empty((0,)), np.empty((0,))

    feat_arr  = features.loc[common].values.astype(np.float32)
    label_arr = labels.loc[common].values.astype(np.int64)

    # Expanding z-score (no leakage)
    cumsum  = np.cumsum(feat_arr, axis=0)
    cumsum2 = np.cumsum(feat_arr ** 2, axis=0)
    counts  = np.arange(1, len(feat_arr) + 1).reshape(-1, 1)
    means   = cumsum / counts
    vars    = cumsum2 / counts - means ** 2
    stds    = np.sqrt(np.maximum(vars, 0))
    stds[stds == 0] = 1.0
    norm = np.nan_to_num((feat_arr - means) / stds)

    X_list, y_list = [], []
    for i in range(SEQ_LEN, len(norm)):
        X_list.append(norm[i - SEQ_LEN:i])
        y_list.append(label_arr[i])

    if not X_list:
        return np.empty((0,)), np.empty((0,))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def train_fold(model: TCNPredictor, X_tr, y_tr, X_val, y_val,
               device: torch.device, class_weights: np.ndarray = None) -> float:
    """Train one fold with class-weighted loss. Returns per-class accuracy."""
    # Weighted cross-entropy: without this, model learns to always predict
    # majority class (class 2 = ~72%) and reports 72% accuracy with zero signal.
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    X_t = torch.tensor(X_tr, device=device)
    y_t = torch.tensor(y_tr, device=device)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True)

    X_v = torch.tensor(X_val, device=device)
    y_v = torch.tensor(y_val, device=device)

    best_val = 0.0
    no_improve = 0

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_v).argmax(1)
            acc   = float((preds == y_v).float().mean())

        if acc > best_val:
            best_val   = acc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    return best_val


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    log.info("Device: %s", device)

    # 1. Download
    price_data = download_universe()
    spy_prices = price_data.get("SPY")
    if spy_prices is None:
        log.error("SPY data unavailable. Aborting.")
        return

    # 2. Build pooled feature dataset
    log.info("Building features for all symbols...")
    all_feats  = []
    all_labels = []

    for sym, df in price_data.items():
        feats = build_features_for_symbol(df, sym)
        if feats.empty:
            log.warning("  %s: insufficient data, skip", sym)
            continue

        # FIX: Use each symbol's OWN prices for regime labels (not SPY for everything).
        # SPY in mean-reversion ≠ TSLA in mean-reversion.
        sym_close = df["Close"].squeeze().reindex(feats.index).ffill()
        labels = label_regimes(sym_close)

        pure = feats.drop(columns=["_symbol"])
        common = pure.index.intersection(labels.index)
        if len(common) < 100:
            continue

        all_feats.append(pure.loc[common])
        all_labels.append(labels.loc[common])
        log.info("  %s: %d rows", sym, len(common))

    if not all_feats:
        log.error("No features built. Aborting.")
        return

    features_df = pd.concat(all_feats)
    labels_s    = pd.concat(all_labels)
    n_features  = len(FEAT_COLS)
    # Use actual features available
    available = [c for c in FEAT_COLS if c in features_df.columns]
    features_df = features_df[available]
    n_features  = len(available)

    log.info("Pooled: %d rows, %d features, %d symbols",
             len(features_df), n_features, len(all_feats))

    # Class distribution + inverse-frequency weights (CRITICAL fix for imbalance)
    dist = labels_s.value_counts().sort_index()
    log.info("Class distribution: %s", dict(dist))
    class_wts = compute_class_weights(labels_s, num_classes=4)
    log.info("Class weights (inv-freq): %s", {i: round(float(w), 3) for i, w in enumerate(class_wts)})

    # 3. Walk-forward training
    log.info("Starting walk-forward training...")
    unique_dates = sorted(features_df.index.unique())
    n_dates      = len(unique_dates)

    fold_metrics = []
    best_acc     = 0.0
    best_model   = None

    cursor = TRAIN_WINDOW
    fold   = 0

    while cursor + PURGE + TEST_WINDOW <= n_dates:
        train_dates = set(unique_dates[max(0, cursor - TRAIN_WINDOW):cursor])
        test_start  = cursor + PURGE
        test_end    = min(test_start + TEST_WINDOW, n_dates)
        test_dates  = set(unique_dates[test_start:test_end])

        train_mask = features_df.index.isin(train_dates)
        test_mask  = features_df.index.isin(test_dates)

        X_tr, y_tr = build_sequences(
            features_df.loc[train_mask], labels_s.loc[labels_s.index.isin(train_dates)]
        )
        X_te, y_te = build_sequences(
            features_df.loc[test_mask], labels_s.loc[labels_s.index.isin(test_dates)]
        )

        if len(X_tr) < 100 or len(X_te) == 0:
            log.warning("Fold %d: insufficient data, skip", fold)
            cursor += TEST_WINDOW
            fold   += 1
            continue

        log.info("Fold %d: %d train seqs, %d test seqs",
                 fold, len(X_tr), len(X_te))

        model = TCNPredictor(
            input_size=n_features,
            num_channels=[64, 64, 32],
            kernel_size=3,
            dropout=DROPOUT,
            num_classes=4,
        ).to(device)

        val_acc = train_fold(model, X_tr, y_tr, X_te, y_te, device, class_weights=class_wts)
        log.info("  Fold %d accuracy: %.4f", fold, val_acc)

        fold_metrics.append({
            "fold": fold,
            "val_acc": round(val_acc, 4),
            "train_seqs": int(len(X_tr)),
            "test_seqs": int(len(X_te)),
        })

        # Save best model
        ckpt_path = str(MODEL_DIR / f"tcn_fold{fold}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "feature_names": available,
            "val_acc": val_acc,
            "fold": fold,
        }, ckpt_path)

        # Also save normalization statistics so inference can use IDENTICAL normalization
        # to what training saw. Without this, the model sees out-of-distribution inputs.
        train_feats_arr = features_df.loc[features_df.index.isin(train_dates)][available].values.astype(np.float32)
        feat_mean = train_feats_arr.mean(axis=0)
        feat_std  = train_feats_arr.std(axis=0)
        feat_std[feat_std == 0] = 1.0

        if val_acc > best_acc:
            best_acc  = val_acc
            best_model = ckpt_path
            torch.save({
                "state_dict": model.state_dict(),
                "n_features": n_features,
                "feature_names": available,
                "val_acc": val_acc,
                "fold": fold,
                "feat_mean": feat_mean,   # CRITICAL: save for consistent inference
                "feat_std": feat_std,     # CRITICAL: save for consistent inference
            }, str(MODEL_DIR / "tcn_tda_model.pt"))
            log.info("  ★ New best model (acc=%.4f)", best_acc)

        cursor += TEST_WINDOW
        fold   += 1

    # 4. Report
    print("\n" + "=" * 60)
    print("  TCN + TDA TRAINING RESULTS")
    print("=" * 60)
    print(f"  Folds:         {len(fold_metrics)}")
    if fold_metrics:
        accs = [f["val_acc"] for f in fold_metrics]
        print(f"  Avg Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Best Accuracy: {max(accs):.4f}")
        print(f"  Min Accuracy:  {min(accs):.4f}")
        print(f"  (Random baseline: 0.25 for 4 classes)")
    print(f"  Best model:    {best_model}")
    print(f"  Time:          {(time.time()-t0)/60:.1f} min")
    print("=" * 60)

    # Save results
    result = {
        "timestamp": datetime.now().isoformat(),
        "folds": fold_metrics,
        "best_accuracy": float(best_acc),
        "n_features": n_features,
        "feature_names": available,
        "best_model": best_model,
    }
    with open(str(RESULT_DIR / "training_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    log.info("Training complete.")


if __name__ == "__main__":
    main()
