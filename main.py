"""Main integration script: Train NN on historical data and run backtest.

2025 Validation Version:
- Downloads SPY data 2022-2025 via yfinance
- Trains NN on 2022 data only
- Runs 3 backtests: 2023-2024, 2025 only, 2023-2025 combined
- Outputs validation results to backtest_2025_validation.json
- DIAGNOSTIC VERSION: Includes NN training debug instrumentation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Diagnostic log file path
NN_DEBUG_LOG = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/nn_training_debug.txt'


def write_debug_log(message: str, also_print: bool = True):
    """Write message to debug log file and optionally print."""
    os.makedirs(os.path.dirname(NN_DEBUG_LOG), exist_ok=True)
    with open(NN_DEBUG_LOG, 'a') as f:
        f.write(message + '\n')
    if also_print:
        print(message)


def get_model_weight_sample(model, layer_name: str = "lstm1") -> str:
    """Get sample values from a model layer's weights."""
    try:
        for layer in model.layers:
            if layer_name in layer.name.lower():
                weights = layer.get_weights()
                if weights:
                    w = weights[0]  # First weight matrix
                    flat = w.flatten()
                    sample = flat[:10] if len(flat) >= 10 else flat
                    return f"[{', '.join(f'{v:.6f}' for v in sample)}]"
        return "[No weights found]"
    except Exception as e:
        return f"[Error: {e}]"


def get_model_weight_stats(model) -> dict:
    """Get statistics about all model weights."""
    all_weights = []
    for layer in model.layers:
        for w in layer.get_weights():
            all_weights.extend(w.flatten())
    
    if not all_weights:
        return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'zeros': 0}
    
    arr = np.array(all_weights)
    return {
        'count': len(arr),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'zeros': int(np.sum(np.abs(arr) < 1e-10))
    }


def run_inference_test(model, n_features: int = 6, sequence_length: int = 15, 
                       n_tests: int = 10) -> list:
    """Run inference on random inputs to check output variability."""
    results = []
    for i in range(n_tests):
        np.random.seed(i * 42)
        test_input = np.random.randn(1, sequence_length, n_features).astype('float32')
        output = model(test_input, training=False)
        results.append(float(output.numpy()[0, 0]))
    return results


def print_training_diagnostics(X_train: np.ndarray, y_train: np.ndarray, 
                                model, history, weights_path: str):
    """Print comprehensive NN training diagnostics."""
    
    # Clear previous log
    with open(NN_DEBUG_LOG, 'w') as f:
        f.write(f"NN Training Diagnostics - {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
    
    write_debug_log("\n" + "═" * 60)
    write_debug_log("NN TRAINING DIAGNOSTICS")
    write_debug_log("═" * 60)
    
    # ---- Training Data Statistics ----
    write_debug_log("\n[1] TRAINING DATA:")
    write_debug_log(f"    X_train shape: {X_train.shape}")
    write_debug_log(f"    X_train dtype: {X_train.dtype}")
    write_debug_log(f"    X_train mean:  {np.mean(X_train):.6f}")
    write_debug_log(f"    X_train std:   {np.std(X_train):.6f}")
    write_debug_log(f"    X_train min:   {np.min(X_train):.6f}")
    write_debug_log(f"    X_train max:   {np.max(X_train):.6f}")
    write_debug_log(f"    X_train NaN:   {np.sum(np.isnan(X_train))}")
    write_debug_log(f"    X_train Inf:   {np.sum(np.isinf(X_train))}")
    
    write_debug_log(f"\n    y_train shape: {y_train.shape}")
    write_debug_log(f"    y_train dtype: {y_train.dtype}")
    write_debug_log(f"    y_train mean:  {np.mean(y_train):.4f} (should be ~0.5 for balanced)")
    write_debug_log(f"    y_train sum:   {np.sum(y_train)} (class 1 count)")
    write_debug_log(f"    y_train class balance: {np.mean(y_train)*100:.1f}% up, {(1-np.mean(y_train))*100:.1f}% down")
    
    # ---- Training History ----
    write_debug_log("\n[2] TRAINING HISTORY:")
    if history and hasattr(history, 'history'):
        loss_hist = history.history.get('loss', [])
        val_loss_hist = history.history.get('val_loss', [])
        acc_hist = history.history.get('accuracy', [])
        val_acc_hist = history.history.get('val_accuracy', [])
        
        write_debug_log(f"    Epochs completed: {len(loss_hist)}")
        
        if loss_hist:
            write_debug_log(f"    Initial loss:    {loss_hist[0]:.6f}")
            write_debug_log(f"    Final loss:      {loss_hist[-1]:.6f}")
            write_debug_log(f"    Loss reduction:  {loss_hist[0] - loss_hist[-1]:.6f}")
            
            if loss_hist[-1] >= loss_hist[0] - 0.001:
                write_debug_log("    ⚠️  WARNING: Loss did NOT decrease (training may have failed)")
            else:
                write_debug_log("    ✓ Loss decreased during training")
        
        if val_loss_hist:
            write_debug_log(f"    Initial val_loss: {val_loss_hist[0]:.6f}")
            write_debug_log(f"    Final val_loss:   {val_loss_hist[-1]:.6f}")
        
        if acc_hist:
            write_debug_log(f"    Initial accuracy: {acc_hist[0]*100:.2f}%")
            write_debug_log(f"    Final accuracy:   {acc_hist[-1]*100:.2f}%")
        
        if val_acc_hist:
            write_debug_log(f"    Final val_acc:    {val_acc_hist[-1]*100:.2f}%")
    else:
        write_debug_log("    ⚠️  No training history available!")
    
    # ---- Model Weights ----
    write_debug_log("\n[3] MODEL WEIGHTS (Post-Training):")
    weight_stats = get_model_weight_stats(model)
    write_debug_log(f"    Total params:    {weight_stats['count']}")
    write_debug_log(f"    Weights mean:    {weight_stats['mean']:.6f}")
    write_debug_log(f"    Weights std:     {weight_stats['std']:.6f}")
    write_debug_log(f"    Weights min:     {weight_stats['min']:.6f}")
    write_debug_log(f"    Weights max:     {weight_stats['max']:.6f}")
    write_debug_log(f"    Zero weights:    {weight_stats['zeros']} ({weight_stats['zeros']/max(weight_stats['count'],1)*100:.1f}%)")
    
    lstm_sample = get_model_weight_sample(model, "lstm")
    write_debug_log(f"    LSTM weight sample: {lstm_sample}")
    
    if weight_stats['std'] < 0.001:
        write_debug_log("    ⚠️  WARNING: Weight std is near zero (untrained network?)")
    else:
        write_debug_log("    ✓ Weights have reasonable variance")
    
    # ---- Saved Checkpoint ----
    write_debug_log("\n[4] MODEL CHECKPOINT:")
    write_debug_log(f"    Path: {weights_path}")
    
    if os.path.exists(weights_path):
        file_size = os.path.getsize(weights_path)
        file_size_mb = file_size / (1024 * 1024)
        write_debug_log(f"    Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
        
        if file_size_mb < 0.1:
            write_debug_log("    ⚠️  WARNING: File is suspiciously small (< 0.1 MB)")
        else:
            write_debug_log("    ✓ Checkpoint file size looks reasonable")
    else:
        write_debug_log("    ⚠️  WARNING: Checkpoint file not found!")
    
    # ---- Post-Training Inference Test ----
    write_debug_log("\n[5] POST-TRAINING INFERENCE TEST:")
    inference_results = run_inference_test(model, n_features=6, sequence_length=15, n_tests=10)
    
    write_debug_log(f"    10 random inputs → outputs:")
    for i, result in enumerate(inference_results):
        write_debug_log(f"      Test {i+1}: {result:.6f}")
    
    mean_output = np.mean(inference_results)
    std_output = np.std(inference_results)
    write_debug_log(f"\n    Output mean: {mean_output:.6f}")
    write_debug_log(f"    Output std:  {std_output:.6f}")
    
    # Check for constant 0.5 output
    if all(abs(r - 0.5) < 0.01 for r in inference_results):
        write_debug_log("    ❌ CRITICAL: All outputs are ~0.5 (UNTRAINED NETWORK BEHAVIOR)")
        write_debug_log("       This means the model learned nothing or weights are zero.")
    elif std_output < 0.01:
        write_debug_log(f"    ⚠️  WARNING: Output variance is very low ({std_output:.6f})")
        write_debug_log("       Model may not be learning directional signals.")
    else:
        write_debug_log("    ✓ Outputs vary across different inputs (model is trained)")
    
    # ---- On Actual Training Data ----
    write_debug_log("\n[6] INFERENCE ON ACTUAL TRAINING DATA:")
    n_samples = min(20, len(X_train))
    sample_indices = np.linspace(0, len(X_train)-1, n_samples, dtype=int)
    
    train_outputs = []
    for idx in sample_indices:
        X_sample = X_train[idx:idx+1]
        output = model(X_sample, training=False)
        train_outputs.append(float(output.numpy()[0, 0]))
    
    write_debug_log(f"    {n_samples} training samples → outputs:")
    for i, (idx, out) in enumerate(zip(sample_indices, train_outputs)):
        actual = int(y_train[idx])
        pred_class = 1 if out >= 0.5 else 0
        match = "✓" if pred_class == actual else "✗"
        write_debug_log(f"      Sample {idx:3d}: output={out:.4f}, actual={actual}, pred={pred_class} {match}")
    
    train_mean = np.mean(train_outputs)
    train_std = np.std(train_outputs)
    write_debug_log(f"\n    Training data output mean: {train_mean:.6f}")
    write_debug_log(f"    Training data output std:  {train_std:.6f}")
    
    if all(abs(r - 0.5) < 0.02 for r in train_outputs):
        write_debug_log("    ❌ CRITICAL: All training outputs are ~0.5!")
        write_debug_log("       Model is NOT learning from the training data.")
    
    # ---- Summary Verdict ----
    write_debug_log("\n" + "═" * 60)
    write_debug_log("DIAGNOSTIC VERDICT:")
    write_debug_log("═" * 60)
    
    issues = []
    
    if history and hasattr(history, 'history'):
        loss_hist = history.history.get('loss', [])
        if loss_hist and loss_hist[-1] >= loss_hist[0] - 0.001:
            issues.append("Training loss did not decrease")
    
    if weight_stats['std'] < 0.001:
        issues.append("Model weights have near-zero variance")
    
    if all(abs(r - 0.5) < 0.01 for r in inference_results):
        issues.append("Model outputs constant 0.5 (untrained)")
    
    if os.path.exists(weights_path) and os.path.getsize(weights_path) < 100000:
        issues.append("Checkpoint file is suspiciously small")
    
    if issues:
        write_debug_log("❌ ISSUES DETECTED:")
        for issue in issues:
            write_debug_log(f"   • {issue}")
        write_debug_log("\nThe NN model appears to be BROKEN or UNTRAINED.")
        write_debug_log("Possible causes:")
        write_debug_log("  1. Training data (X_train, y_train) is malformed")
        write_debug_log("  2. Model.fit() didn't run or learn anything")
        write_debug_log("  3. Learning rate too low / epochs too few")
        write_debug_log("  4. Features are all zeros or NaN")
    else:
        write_debug_log("✓ No obvious issues detected. Model appears trained.")
        write_debug_log("  If outputs are still ~0.5 during backtest, check:")
        write_debug_log("  1. Is the correct model being loaded?")
        write_debug_log("  2. Are features computed the same way during inference?")
    
    write_debug_log("═" * 60 + "\n")
    write_debug_log(f"Full diagnostics saved to: {NN_DEBUG_LOG}")
    
    return issues

from src.tda_features import TDAFeatureGenerator
from src.nn_predictor import NeuralNetPredictor, DataPreprocessor, train_model, OutputSpreadCallback
from src.ensemble_strategy import EnsembleStrategy, PerformanceAnalyzer

import backtrader as bt

# Entropy penalty analysis log
ENTROPY_ANALYSIS_LOG = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/entropy_penalty_analysis.txt'


def write_entropy_log(message: str, also_print: bool = True):
    """Write message to entropy analysis log file."""
    os.makedirs(os.path.dirname(ENTROPY_ANALYSIS_LOG), exist_ok=True)
    with open(ENTROPY_ANALYSIS_LOG, 'a') as f:
        f.write(message + '\n')
    if also_print:
        print(message)


def compare_output_spread(model, X_data: np.ndarray, label: str) -> dict:
    """Analyze model output distribution on given data."""
    predictions = model.predict(X_data, verbose=0).flatten()
    
    stats = {
        'label': label,
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'pct_above_055': float(np.mean(predictions > 0.55) * 100),
        'pct_below_045': float(np.mean(predictions < 0.45) * 100),
        'pct_extreme': float(np.mean((predictions > 0.55) | (predictions < 0.45)) * 100),
        'pct_above_060': float(np.mean(predictions > 0.60) * 100),
        'pct_below_040': float(np.mean(predictions < 0.40) * 100),
    }
    return stats


def print_spread_comparison(baseline_stats: dict, entropy_stats: dict):
    """Print before/after comparison of output spread."""
    write_entropy_log("\n" + "═" * 70)
    write_entropy_log("ENTROPY PENALTY ANALYSIS: BEFORE vs AFTER")
    write_entropy_log("═" * 70)
    
    write_entropy_log(f"\n{'Metric':<25} {'Baseline (BCE)':<20} {'Entropy Penalty':<20} {'Change':<15}")
    write_entropy_log("-" * 70)
    
    for metric in ['mean', 'std', 'min', 'max']:
        baseline = baseline_stats[metric]
        entropy = entropy_stats[metric]
        change = entropy - baseline
        write_entropy_log(f"{metric:<25} {baseline:<20.4f} {entropy:<20.4f} {change:+.4f}")
    
    write_entropy_log("-" * 70)
    
    for metric in ['pct_above_055', 'pct_below_045', 'pct_extreme', 'pct_above_060', 'pct_below_040']:
        baseline = baseline_stats[metric]
        entropy = entropy_stats[metric]
        change = entropy - baseline
        label = metric.replace('pct_', '% ').replace('_', ' ')
        write_entropy_log(f"{label:<25} {baseline:<19.1f}% {entropy:<19.1f}% {change:+.1f}%")
    
    write_entropy_log("-" * 70)
    
    # Verdict
    write_entropy_log("\n[VERDICT]:")
    if entropy_stats['std'] > baseline_stats['std'] * 1.5:
        write_entropy_log("✅ SUCCESS: Entropy penalty significantly increased output variance!")
        write_entropy_log(f"   std: {baseline_stats['std']:.4f} → {entropy_stats['std']:.4f} ({entropy_stats['std']/baseline_stats['std']:.1f}x increase)")
    elif entropy_stats['std'] > baseline_stats['std'] * 1.1:
        write_entropy_log("⚠️  PARTIAL: Entropy penalty modestly increased output variance.")
        write_entropy_log("   Consider increasing entropy_weight from 0.05 to 0.10")
    else:
        write_entropy_log("❌ MINIMAL EFFECT: Entropy penalty did not significantly spread outputs.")
        write_entropy_log("   This suggests the financial data itself lacks predictive signal.")
        write_entropy_log("   Consider: adding more features, or accepting NN as secondary filter.")
    
    if entropy_stats['pct_extreme'] > 30:
        write_entropy_log(f"✅ TRADEABLE: {entropy_stats['pct_extreme']:.1f}% of predictions are in extreme zones (>0.55 or <0.45)")
    elif entropy_stats['pct_extreme'] > 15:
        write_entropy_log(f"⚠️  MARGINAL: {entropy_stats['pct_extreme']:.1f}% in extreme zones - may need threshold adjustment")
    else:
        write_entropy_log(f"❌ LOW SIGNAL: Only {entropy_stats['pct_extreme']:.1f}% in extreme zones")
    
    write_entropy_log("═" * 70)


def download_spy_data(start_date: str = '2022-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """Download SPY data from Yahoo Finance."""
    print(f"  Downloading SPY data from {start_date} to {end_date}...")
    ticker = yf.Ticker('SPY')
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Remove timezone info from index for compatibility
    df.index = df.index.tz_localize(None)
    
    print(f"  Downloaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def split_data_by_year(df: pd.DataFrame) -> dict:
    """Split data into yearly segments for targeted backtesting."""
    return {
        '2022': df[df.index.year == 2022].copy(),
        '2023': df[df.index.year == 2023].copy(),
        '2024': df[df.index.year == 2024].copy(),
        '2025': df[df.index.year == 2025].copy(),
        '2023_2024': df[(df.index.year >= 2023) & (df.index.year <= 2024)].copy(),
        '2023_2025': df[(df.index.year >= 2023) & (df.index.year <= 2025)].copy(),
    }


def prepare_training_data(train_df: pd.DataFrame, tda_gen: TDAFeatureGenerator,
                          preprocessor: DataPreprocessor) -> tuple:
    """Generate TDA features and prepare sequences for training."""
    tda_features = tda_gen.generate_features(train_df)
    X, y = preprocessor.prepare_sequences(train_df, tda_features)
    return X.astype(np.float32), y.astype(np.float32)


def run_backtest(test_df: pd.DataFrame, model: NeuralNetPredictor,
                 tda_gen: TDAFeatureGenerator, preprocessor: DataPreprocessor,
                 initial_cash: float = 100000, label: str = "Backtest") -> dict:
    """Run backtest with trained model and return performance metrics.
    
    Creates a fresh cerebro instance each time to avoid state carryover.
    """
    print(f"  Running {label} ({len(test_df)} bars: {test_df.index[0].date()} to {test_df.index[-1].date()})...")
    
    cerebro = bt.Cerebro()
    
    data = bt.feeds.PandasData(dataname=test_df)
    cerebro.adddata(data)
    
    cerebro.addstrategy(
        EnsembleStrategy,
        nn_model=model,
        tda_generator=tda_gen,
        preprocessor=preprocessor,
        sequence_length=15,
        verbose=False
    )
    
    cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)
    
    results = cerebro.run()
    
    metrics = extract_metrics(results, initial_cash, cerebro.broker.getvalue())
    print(f"    Sharpe: {metrics['sharpe_ratio']:.4f}, Trades: {metrics['num_trades']}, Return: {metrics['total_return']*100:.2f}%")
    
    return metrics


def extract_metrics(results, initial_cash: float, final_value: float) -> dict:
    """Extract performance metrics from backtest results."""
    strat = results[0]
    
    returns_dict = strat.analyzers.returns.get_analysis()
    daily_returns = list(returns_dict.values()) if returns_dict else []
    
    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0
    
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.get('max', {}).get('drawdown', 0) or 0
    
    perf = strat.analyzers.performance.get_analysis()
    
    total_return = (final_value - initial_cash) / initial_cash
    
    avg_win = perf.get('avg_win', 0)
    avg_loss = abs(perf.get('avg_loss', 1)) or 1
    
    return {
        'sharpe_ratio': round(float(sharpe), 4),
        'max_drawdown': round(float(max_dd) / 100, 4),
        'total_return': round(float(total_return), 4),
        'num_trades': int(perf.get('num_trades', 0)),
        'win_rate': round(float(perf.get('win_rate', 0)), 4),
        'avg_win_loss': round(float(avg_win / avg_loss), 4),
        'final_value': round(float(final_value), 2),
        'initial_cash': float(initial_cash)
    }


def save_results(metrics: dict, filepath: str):
    """Save backtest results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_validation_summary(results: dict):
    """Print formatted summary comparing all backtest periods."""
    print("\n")
    print("═" * 60)
    print("OUT-OF-SAMPLE VALIDATION: 2025 TEST")
    print("═" * 60)
    
    sharpe_2023_2024 = results['backtest_2023_2024']['sharpe_ratio']
    sharpe_2025 = results['backtest_2025_only']['sharpe_ratio']
    sharpe_full = results['backtest_2023_2025']['sharpe_ratio']
    
    print(f"2023-2024 (Original):    Sharpe = {sharpe_2023_2024:.2f}")
    print(f"2025 Only (New OOS):     Sharpe = {sharpe_2025:.2f}")
    print(f"2023-2025 (Full):        Sharpe = {sharpe_full:.2f}")
    print("-" * 60)
    
    # Calculate degradation
    if sharpe_2023_2024 > 0:
        degradation = (sharpe_2023_2024 - sharpe_2025) / sharpe_2023_2024 * 100
    else:
        degradation = 0
    
    print(f"Sharpe Change (2025 vs 2023-2024): {sharpe_2025 - sharpe_2023_2024:+.2f} ({-degradation:+.1f}%)")
    print("-" * 60)
    
    # Verdict
    if sharpe_2025 >= 1.5:
        verdict = "✅ ROBUST - Model generalized well to unseen 2025 data!"
        verdict_short = "ROBUST"
    elif sharpe_2025 >= 1.0:
        verdict = "⚠️  MARGINAL - Sharpe degraded but still > 1.0"
        verdict_short = "MARGINAL"
    else:
        verdict = "❌ OVERFIT - Model failed on out-of-sample 2025 data"
        verdict_short = "OVERFIT"
    
    print(f"\nVerdict: [{verdict_short}]")
    print(verdict)
    print("═" * 60)
    
    # Detailed metrics table
    print("\nDETAILED METRICS:")
    print("-" * 60)
    print(f"{'Period':<20} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8}")
    print("-" * 60)
    
    for period_key, label in [('backtest_2023_2024', '2023-2024'), 
                               ('backtest_2025_only', '2025 Only'),
                               ('backtest_2023_2025', '2023-2025')]:
        m = results[period_key]
        print(f"{label:<20} {m['sharpe_ratio']:>8.2f} {m['total_return']*100:>9.2f}% {m['max_drawdown']*100:>7.2f}% {m['num_trades']:>8} {m['win_rate']*100:>7.1f}%")
    
    print("-" * 60)


def main():
    """Main execution: download data, train model on 2022, run 3 backtests."""
    results_path = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/backtest_2025_validation.json'
    
    # =========================================================================
    # STEP 1: Download full SPY data 2022-2025
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Downloading SPY Data (2022-2025)")
    print("=" * 60)
    df = download_spy_data(start_date='2022-01-01', end_date='2025-12-31')
    
    # Split by year
    data_splits = split_data_by_year(df)
    print(f"\nData splits:")
    for key, split_df in data_splits.items():
        if len(split_df) > 0:
            print(f"  {key}: {len(split_df)} bars")
    
    # =========================================================================
    # STEP 2: Train NN on 2022 data ONLY (with entropy penalty comparison)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Training Neural Network (2022 Data Only)")
    print("=" * 60)
    
    train_df = data_splits['2022']
    print(f"  Training period: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} bars)")
    
    print("  Generating TDA features...")
    tda_gen = TDAFeatureGenerator(window=20, embedding_dim=3)
    preprocessor = DataPreprocessor(sequence_length=15)
    
    X_train, y_train = prepare_training_data(train_df, tda_gen, preprocessor)
    print(f"  Training samples: {len(X_train)}")
    
    # Quick data sanity check
    print(f"  X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"  y_train class balance: {np.mean(y_train)*100:.1f}% up, {(1-np.mean(y_train))*100:.1f}% down")
    
    # Clear entropy analysis log
    with open(ENTROPY_ANALYSIS_LOG, 'w') as f:
        f.write(f"Entropy Penalty Analysis - {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
    
    # -------------------------------------------------------------------------
    # PHASE A: Train with STANDARD binary crossentropy (baseline)
    # -------------------------------------------------------------------------
    print("\n  [A] Training with STANDARD binary crossentropy (baseline)...")
    model_baseline = NeuralNetPredictor(sequence_length=15, n_features=6, lstm_units=64)
    model_baseline.compile_model(learning_rate=0.0005, use_entropy_penalty=False)
    
    # Build model
    _ = model_baseline(X_train[:1])
    
    # Train without spread tracking (faster)
    history_baseline = train_model(model_baseline, X_train, y_train, epochs=100, batch_size=16,
                                    track_output_spread=False)
    
    # Get baseline output statistics
    baseline_stats = compare_output_spread(model_baseline, X_train, "Baseline (BCE)")
    print(f"      Baseline output: mean={baseline_stats['mean']:.4f}, std={baseline_stats['std']:.4f}, "
          f"extreme={baseline_stats['pct_extreme']:.1f}%")
    
    # -------------------------------------------------------------------------
    # PHASE B: Train with ENTROPY PENALTY (experimental)
    # -------------------------------------------------------------------------
    print("\n  [B] Training with ENTROPY PENALTY loss (experimental)...")
    model_entropy = NeuralNetPredictor(sequence_length=15, n_features=6, lstm_units=64)
    model_entropy.compile_model(learning_rate=0.0005, use_entropy_penalty=True, entropy_weight=0.05)
    
    # Build model
    _ = model_entropy(X_train[:1])
    
    # Train WITH spread tracking
    print("      Tracking output spread per epoch:")
    history_entropy, spread_callback = train_model(
        model_entropy, X_train, y_train, epochs=100, batch_size=16,
        track_output_spread=True, verbose_spread=True
    )
    
    # Get entropy penalty output statistics
    entropy_stats = compare_output_spread(model_entropy, X_train, "Entropy Penalty")
    print(f"      Entropy output: mean={entropy_stats['mean']:.4f}, std={entropy_stats['std']:.4f}, "
          f"extreme={entropy_stats['pct_extreme']:.1f}%")
    
    # -------------------------------------------------------------------------
    # Compare baseline vs entropy penalty
    # -------------------------------------------------------------------------
    print_spread_comparison(baseline_stats, entropy_stats)
    
    # Choose the better model based on output spread
    if entropy_stats['std'] > baseline_stats['std'] * 1.1:
        print("\n  → Using ENTROPY PENALTY model (better output spread)")
        model = model_entropy
        history = history_entropy
    else:
        print("\n  → Using BASELINE model (entropy penalty didn't help)")
        model = model_baseline
        history = history_baseline
    
    # Save model weights
    weights_path = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/model_weights.weights.h5'
    model.save_checkpoint(weights_path)
    print(f"  Model saved to {weights_path}")
    
    # =========================================================================
    # STEP 2.5: Run NN Training Diagnostics
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2.5: NN Training Diagnostics")
    print("=" * 60)
    diagnostic_issues = print_training_diagnostics(X_train, y_train, model, history, weights_path)
    
    if diagnostic_issues:
        print("\n⚠️  DIAGNOSTIC ISSUES FOUND - Review above before trusting backtest results!")
    
    # =========================================================================
    # STEP 3: Run THREE separate backtests
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Running Backtests (3 Separate Periods)")
    print("=" * 60)
    
    validation_results = {}
    
    # Backtest 1: 2023-2024 (Original benchmark)
    print("\n[1/3] Backtest: 2023-2024 (Original Period)")
    metrics_2023_2024 = run_backtest(
        data_splits['2023_2024'], model, tda_gen, preprocessor,
        label="2023-2024"
    )
    validation_results['backtest_2023_2024'] = metrics_2023_2024
    
    # Backtest 2: 2025 Only (New out-of-sample test)
    print("\n[2/3] Backtest: 2025 Only (Out-of-Sample)")
    metrics_2025 = run_backtest(
        data_splits['2025'], model, tda_gen, preprocessor,
        label="2025 Only"
    )
    validation_results['backtest_2025_only'] = metrics_2025
    
    # Backtest 3: 2023-2025 Combined (Full 3-year walk-forward)
    print("\n[3/3] Backtest: 2023-2025 Combined (Full Period)")
    metrics_full = run_backtest(
        data_splits['2023_2025'], model, tda_gen, preprocessor,
        label="2023-2025"
    )
    validation_results['backtest_2023_2025'] = metrics_full
    
    # =========================================================================
    # STEP 4: Save and display results
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Saving Results")
    print("=" * 60)
    
    save_results(validation_results, results_path)
    print(f"  Results saved to: {results_path}")
    
    # Print summary comparison
    print_validation_summary(validation_results)
    
    return validation_results


if __name__ == "__main__":
    results = main()
