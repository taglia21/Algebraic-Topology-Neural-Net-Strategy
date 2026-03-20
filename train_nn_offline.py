#!/usr/bin/env python3
"""
train_nn_offline.py
===================
Offline NN training script — fetches historical data via yfinance,
builds features, runs walk-forward training, and saves the best model.

Run this standalone (no IBKR connection needed):
    cd /path/to/atnn-bot
    python train_nn_offline.py

The trained model is saved to ./models/ (or the configured model_dir).
"""

import sys
import os
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nn.features import NNFeatureEngine
from nn.data_loader import direction_labels
from nn.training import WalkForwardTrainer
from nn.models.lstm_predictor import LSTMPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_nn_offline")

# ── Configuration ──────────────────────────────────────────────────────
SYMBOLS = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "XLF", "XLE", "XLK", "XLV", "XLI",
    "GLD", "TLT",
]

# Training config (matches live.yaml)
TRAIN_WINDOW = 756       # ~3 years
TEST_WINDOW = 21         # 1 month
PURGE_GAP = 5
EMBARGO_GAP = 5
MAX_EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 10
SEQUENCE_LENGTH = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
DIRECTION_THRESHOLD = 0.005
LOOKBACK_YEARS = 5       # fetch 5 years of daily data

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 5 years of daily OHLCV data via yfinance."""
    logger.info(f"Fetching {LOOKBACK_YEARS}Y daily data for {len(SYMBOLS)} symbols...")

    closes = {}
    volumes = {}

    for sym in SYMBOLS:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period=f"{LOOKBACK_YEARS}y", interval="1d")
            if hist is not None and len(hist) > 100:
                closes[sym] = hist["Close"]
                volumes[sym] = hist["Volume"]
                logger.info(f"  {sym}: {len(hist)} bars")
            else:
                logger.warning(f"  {sym}: insufficient data ({len(hist) if hist is not None else 0} bars)")
        except Exception as e:
            logger.warning(f"  {sym}: fetch failed — {e}")

    if not closes:
        raise RuntimeError("No data fetched for any symbol")

    price_df = pd.DataFrame(closes)
    volume_df = pd.DataFrame(volumes)

    # Drop any dates where we have too many missing symbols
    min_symbols = len(SYMBOLS) * 0.7  # need at least 70% of symbols
    valid_mask = price_df.notna().sum(axis=1) >= min_symbols
    price_df = price_df[valid_mask]
    volume_df = volume_df[valid_mask]

    # Forward-fill then drop remaining NaNs
    price_df = price_df.ffill().dropna()
    volume_df = volume_df.ffill().dropna()

    # Align indices
    common_idx = price_df.index.intersection(volume_df.index)
    price_df = price_df.loc[common_idx]
    volume_df = volume_df.loc[common_idx]

    logger.info(f"Final dataset: {len(price_df)} days × {price_df.shape[1]} symbols")
    return price_df, volume_df


def build_features_and_targets(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Build feature matrix and direction labels."""
    logger.info("Building NN features...")

    engine = NNFeatureEngine()

    # Try to add TDA features
    tda_features = None
    try:
        from tda import TDAFeatureExtractor
        tda_ext = TDAFeatureExtractor(
            ph_window=30,
            corr_window=60,
            diffusion_time=1.0,
        )
        returns_df = price_df.pct_change().dropna()
        tda_features = tda_ext.extract(returns_df)
        logger.info(f"  TDA features: {tda_features.shape[1]} columns")
    except Exception as e:
        logger.warning(f"  TDA features skipped: {e}")

    features = engine.build_features(
        price_df=price_df,
        volume_df=volume_df,
        tda_features_df=tda_features,
    )

    feature_names = list(features.columns)
    input_size = len(feature_names)
    logger.info(f"  Feature matrix: {features.shape[0]} rows × {input_size} features")
    logger.info(f"  Features: {feature_names}")

    # Create direction labels from average cross-sectional return
    returns = price_df.pct_change().dropna()
    avg_returns = returns.mean(axis=1)
    target = direction_labels(avg_returns, threshold=DIRECTION_THRESHOLD)

    # Align features and target
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    # Label distribution
    dist = target.value_counts().sort_index()
    logger.info(f"  Labels — Down(0): {dist.get(0, 0)}, Flat(1): {dist.get(1, 0)}, Up(2): {dist.get(2, 0)}")

    return features, target, input_size


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    input_size: int,
) -> tuple[str, dict]:
    """Run walk-forward training and return best model path + metrics."""
    logger.info("Starting walk-forward training...")
    logger.info(f"  Config: window={TRAIN_WINDOW}, horizon={TEST_WINDOW}, "
                f"epochs={MAX_EPOCHS}, batch={BATCH_SIZE}, lr={LR}")

    trainer = WalkForwardTrainer(
        train_window=TRAIN_WINDOW,
        predict_horizon=TEST_WINDOW,
        purge_gap=PURGE_GAP,
        embargo=EMBARGO_GAP,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=LR,
        batch_size=BATCH_SIZE,
        device="cpu",
        checkpoint_dir=str(MODEL_DIR),
    )

    result = trainer.train_walk_forward(
        features_df=features,
        target=target,
        model_class=LSTMPredictor,
        window=SEQUENCE_LENGTH,
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    # Aggregate metrics
    metrics = {}
    if result.metrics_per_fold:
        accuracies = [m.accuracy for m in result.metrics_per_fold]
        val_losses = [m.val_loss for m in result.metrics_per_fold]

        metrics = {
            "num_folds": len(result.metrics_per_fold),
            "avg_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "avg_val_loss": float(np.mean(val_losses)),
            "best_model_path": result.best_model_path,
            "input_size": input_size,
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"  TRAINING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"  Folds completed: {metrics['num_folds']}")
        logger.info(f"  Avg accuracy:    {metrics['avg_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        logger.info(f"  Min/Max acc:     {metrics['min_accuracy']:.4f} / {metrics['max_accuracy']:.4f}")
        logger.info(f"  Avg val loss:    {metrics['avg_val_loss']:.4f}")
        logger.info(f"  Best model:      {metrics['best_model_path']}")
        logger.info(f"  Input size:      {input_size}")
        logger.info(f"{'='*60}\n")

        # Per-fold detail
        for m in result.metrics_per_fold:
            logger.info(
                f"  Fold {m.fold}: acc={m.accuracy:.4f}, "
                f"train_loss={m.train_loss:.4f}, val_loss={m.val_loss:.4f}"
            )

        # Copy best model to a canonical name
        if result.best_model_path:
            best_src = Path(result.best_model_path)
            canonical = MODEL_DIR / "best_model.pt"
            if best_src.exists():
                import shutil
                shutil.copy2(best_src, canonical)
                logger.info(f"\n  Best model copied to: {canonical}")

                # Also save model metadata
                meta = {
                    "input_size": input_size,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "num_classes": 3,
                    "model_type": "lstm",
                    "sequence_length": SEQUENCE_LENGTH,
                    "train_window": TRAIN_WINDOW,
                    "metrics": metrics,
                    "feature_names": list(features.columns),
                }
                meta_path = MODEL_DIR / "model_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2, default=str)
                logger.info(f"  Model metadata saved to: {meta_path}")
    else:
        logger.warning("Training produced NO folds — dataset may be too small")
        logger.warning(f"  Dataset size: {len(features)} samples")
        logger.warning(f"  Minimum needed: {TRAIN_WINDOW + PURGE_GAP + TEST_WINDOW + SEQUENCE_LENGTH}")

    return result.best_model_path, metrics


def main():
    logger.info("=" * 60)
    logger.info("  ATNN v2 — OFFLINE NN TRAINING")
    logger.info("=" * 60)

    # Step 1: Fetch data
    price_df, volume_df = fetch_data()

    # Step 2: Build features + targets
    features, target, input_size = build_features_and_targets(price_df, volume_df)

    # Step 3: Train
    best_path, metrics = train_model(features, target, input_size)

    if best_path:
        logger.info(f"\nSUCCESS — model saved to {MODEL_DIR}/best_model.pt")
        logger.info(f"Input size for main.py: {input_size}")
        logger.info("Deploy this model to /opt/atnn/models/ on the droplet.")
    else:
        logger.error("FAILED — no model produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
