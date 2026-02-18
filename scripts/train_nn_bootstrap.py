#!/usr/bin/env python3
"""
NN Predictor Bootstrap Training Script
=======================================
Fetches 1 year of daily close data for top S&P 500 symbols,
computes log returns, trains the NeuralNetPredictor LSTM model,
and saves weights to models/nn_predictor_weights.h5.

Usage:
    python scripts/train_nn_bootstrap.py
    python scripts/train_nn_bootstrap.py --epochs 20 --symbols 30
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nn_predictor import NeuralNetPredictor, train_model

# ---------------------------------------------------------------------------
# Top S&P 500 symbols by market cap (diversified across sectors)
# ---------------------------------------------------------------------------
TOP_SP500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "PG", "JNJ", "COST", "HD", "ABBV",
    "MRK", "WMT", "CRM", "BAC", "NFLX", "AMD", "ORCL", "KO", "PEP", "CVX",
    "TMO", "LIN", "ADBE", "ACN", "ABT", "MCD", "CSCO", "WFC", "DHR", "QCOM",
    "INTC", "TXN", "PM", "NEE", "UPS", "AMGN", "GE", "RTX", "LOW", "CAT",
]


def fetch_close_data(symbols: list, period: str = "1y") -> dict:
    """Fetch daily close prices via yfinance. Returns {symbol: np.array}."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    results = {}
    for sym in symbols:
        try:
            data = yf.download(sym, period=period, interval="1d", progress=False)
            if data is not None and len(data) >= 100:
                close = data["Close"].values.flatten().astype(float)
                results[sym] = close
                print(f"  ✓ {sym}: {len(close)} bars")
            else:
                print(f"  ✗ {sym}: insufficient data")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
    return results


def build_training_set(close_dict: dict, seq_length: int = 20, n_features: int = 6):
    """
    Build (X, y) training arrays from close price dict.
    Features per timestep: [log_return, abs_return, vol_5d, vol_10d, momentum_5d, momentum_10d]
    Label: 1 if next-day close > today's close, else 0.
    """
    all_X = []
    all_y = []

    for sym, close in close_dict.items():
        if len(close) < seq_length + 20:
            continue

        # Compute features
        log_returns = np.diff(np.log(close + 1e-10))
        abs_returns = np.abs(log_returns)
        n = len(log_returns)

        # Rolling features
        vol_5d = np.array([np.std(log_returns[max(0, i - 5):i]) if i >= 5 else 0.01 for i in range(n)])
        vol_10d = np.array([np.std(log_returns[max(0, i - 10):i]) if i >= 10 else 0.01 for i in range(n)])
        mom_5d = np.array([np.sum(log_returns[max(0, i - 5):i]) if i >= 5 else 0.0 for i in range(n)])
        mom_10d = np.array([np.sum(log_returns[max(0, i - 10):i]) if i >= 10 else 0.0 for i in range(n)])

        # Stack features: (n, 6)
        features = np.column_stack([log_returns, abs_returns, vol_5d, vol_10d, mom_5d, mom_10d])

        # Normalize per-symbol
        means = features.mean(axis=0)
        stds = features.std(axis=0) + 1e-10
        features = (features - means) / stds

        # Create sequences
        for i in range(seq_length, n - 1):
            X_seq = features[i - seq_length:i]  # shape (seq_length, 6)
            # Label: next bar up?
            label = 1 if close[i + 1] > close[i] else 0
            all_X.append(X_seq)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Bootstrap NN Predictor training")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--symbols", type=int, default=50, help="Number of symbols to use")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "models" / "nn_predictor_weights.h5"),
                        help="Output weights path")
    args = parser.parse_args()

    symbols = TOP_SP500[:args.symbols]
    output_path = args.output

    print("=" * 60)
    print(f"  NN Predictor Bootstrap Training")
    print(f"  Symbols: {len(symbols)}  |  Epochs: {args.epochs}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/4] Fetching 1 year of daily close data...")
    t0 = time.time()
    close_dict = fetch_close_data(symbols)
    print(f"  Fetched {len(close_dict)} symbols in {time.time() - t0:.1f}s")

    if len(close_dict) < 5:
        print("ERROR: Too few symbols with data. Aborting.")
        sys.exit(1)

    # 2. Build training set
    print("\n[2/4] Building training sequences...")
    X, y = build_training_set(close_dict, seq_length=20, n_features=6)
    print(f"  Training set: X={X.shape}, y={y.shape}")
    print(f"  Class balance: {y.mean():.1%} positive ({int(y.sum())}/{len(y)})")

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 3. Train model
    print(f"\n[3/4] Training NeuralNetPredictor for {args.epochs} epochs...")
    model = NeuralNetPredictor(sequence_length=20, n_features=6)
    model.compile_model()

    history = train_model(
        model, X, y,
        epochs=args.epochs,
        batch_size=64,
        validation_split=0.2,
        track_output_spread=False,
    )

    # 4. Save weights
    print(f"\n[4/4] Saving weights to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_checkpoint(output_path)

    # Summary
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history.get("val_loss", [None])[-1]
    final_acc = history.history.get("accuracy", history.history.get("acc", [None]))[-1]
    final_val_acc = history.history.get("val_accuracy", history.history.get("val_acc", [None]))[-1]

    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Epochs completed : {len(history.history['loss'])}")
    print(f"  Final loss       : {final_loss:.4f}")
    print(f"  Final val_loss   : {final_val_loss:.4f}" if final_val_loss else "  Final val_loss   : N/A")
    print(f"  Final accuracy   : {final_acc:.4f}" if final_acc else "  Final accuracy   : N/A")
    print(f"  Final val_acc    : {final_val_acc:.4f}" if final_val_acc else "  Final val_acc    : N/A")
    print(f"  Weights saved to : {output_path}")
    print(f"  File size        : {os.path.getsize(output_path) / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
