#!/usr/bin/env python3
"""
finalize_model.py — Pick the best fold model and create metadata.

We have 9 fold models from walk-forward training. Evaluate each on
a held-out validation set to pick the best one, then save it as
best_model.pt with metadata.
"""

import sys
import os
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nn.features import NNFeatureEngine
from nn.data_loader import direction_labels, TimeSeriesDataset, collate_fn
from nn.models.lstm_predictor import LSTMPredictor
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finalize_model")

SYMBOLS = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "XLF", "XLE", "XLK", "XLV", "XLI",
    "GLD", "TLT",
]

MODEL_DIR = Path("./models")
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
SEQUENCE_LENGTH = 60
DIRECTION_THRESHOLD = 0.005


def main():
    # 1. Fetch recent data for validation (just 1 year)
    logger.info("Fetching 1Y validation data...")
    closes = {}
    volumes = {}
    for sym in SYMBOLS:
        try:
            hist = yf.Ticker(sym).history(period="1y", interval="1d")
            if hist is not None and len(hist) > 50:
                closes[sym] = hist["Close"]
                volumes[sym] = hist["Volume"]
        except:
            pass

    price_df = pd.DataFrame(closes).ffill().dropna()
    volume_df = pd.DataFrame(volumes).ffill().dropna()
    common_idx = price_df.index.intersection(volume_df.index)
    price_df = price_df.loc[common_idx]
    volume_df = volume_df.loc[common_idx]
    logger.info(f"Validation data: {len(price_df)} days × {price_df.shape[1]} symbols")

    # 2. Build features
    engine = NNFeatureEngine()
    tda_features = None
    try:
        from tda import TDAFeatureExtractor
        tda_ext = TDAFeatureExtractor(ph_window=30, corr_window=60, diffusion_time=1.0)
        returns_df = price_df.pct_change().dropna()
        tda_features = tda_ext.extract(returns_df)
    except Exception as e:
        logger.warning(f"TDA features skipped: {e}")

    features = engine.build_features(
        price_df=price_df,
        volume_df=volume_df,
        tda_features_df=tda_features,
    )
    feature_names = list(features.columns)
    input_size = len(feature_names)
    logger.info(f"Features: {input_size} columns — {feature_names}")

    # 3. Labels
    returns = price_df.pct_change().dropna()
    avg_returns = returns.mean(axis=1)
    target = direction_labels(avg_returns, threshold=DIRECTION_THRESHOLD)
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    # 4. Create dataset for evaluation (last 100 samples)
    dataset = TimeSeriesDataset(features, target, window=SEQUENCE_LENGTH, normalize=True)
    n = len(dataset)
    if n < 50:
        logger.error(f"Not enough validation samples: {n}")
        sys.exit(1)

    val_indices = list(range(max(0, n - 100), n))
    val_subset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 5. Evaluate each fold model
    model_files = sorted(MODEL_DIR.glob("model_fold*.pt"))
    logger.info(f"Found {len(model_files)} fold models")

    best_acc = -1
    best_model_file = None
    best_val_loss = float("inf")
    fold_results = []

    criterion = torch.nn.CrossEntropyLoss()

    for mf in model_files:
        try:
            model = LSTMPredictor(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            )
            state = torch.load(mf, map_location="cpu", weights_only=True)

            # Check if input_size matches
            bn_weight_size = state.get("bn.weight", torch.zeros(1)).shape[0]
            if bn_weight_size != input_size:
                logger.warning(f"  {mf.name}: input_size mismatch ({bn_weight_size} vs {input_size}), skipping")
                continue

            model.load_state_dict(state)
            model.eval()

            all_preds = []
            all_targets = []
            val_losses = []

            with torch.no_grad():
                for x_batch, y_batch, lengths in val_loader:
                    logits = model(x_batch, lengths=lengths)
                    loss = criterion(logits, y_batch)
                    val_losses.append(loss.item())
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(y_batch.numpy())

            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            acc = float((all_preds == all_targets).mean())
            avg_loss = float(np.mean(val_losses))

            fold_results.append({
                "file": mf.name,
                "accuracy": acc,
                "val_loss": avg_loss,
            })
            logger.info(f"  {mf.name}: acc={acc:.4f}, val_loss={avg_loss:.4f}")

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_acc = acc
                best_model_file = mf
        except Exception as e:
            logger.warning(f"  {mf.name}: evaluation failed — {e}")

    if best_model_file is None:
        logger.error("No valid model found!")
        sys.exit(1)

    # 6. Copy best model and save metadata
    canonical = MODEL_DIR / "best_model.pt"
    shutil.copy2(best_model_file, canonical)
    logger.info(f"\nBest model: {best_model_file.name} → best_model.pt")
    logger.info(f"  Accuracy: {best_acc:.4f}, Val loss: {best_val_loss:.4f}")

    meta = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "num_classes": 3,
        "model_type": "lstm",
        "sequence_length": SEQUENCE_LENGTH,
        "direction_threshold": DIRECTION_THRESHOLD,
        "feature_names": feature_names,
        "best_fold": best_model_file.name,
        "best_accuracy": best_acc,
        "best_val_loss": best_val_loss,
        "fold_results": fold_results,
        "num_folds_evaluated": len(fold_results),
    }
    meta_path = MODEL_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"  Metadata saved to: {meta_path}")
    logger.info(f"\n  INPUT_SIZE = {input_size}  ← update main.py with this value")


if __name__ == "__main__":
    main()
