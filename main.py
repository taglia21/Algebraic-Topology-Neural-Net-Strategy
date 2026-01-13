"""Main integration script: Train NN on historical data and run backtest."""

import os
import sys
import json
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tda_features import TDAFeatureGenerator
from src.nn_predictor import NeuralNetPredictor, DataPreprocessor, train_model
from src.ensemble_strategy import EnsembleStrategy, PerformanceAnalyzer

import backtrader as bt


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load OHLCV data from CSV or generate synthetic data for testing."""
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
        df.columns = [c.lower() for c in df.columns]
        return df
    
    return generate_synthetic_data()


def generate_synthetic_data(n_bars: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic SPY-like price data with realistic patterns."""
    np.random.seed(seed)
    
    dates = pd.date_range('2022-01-01', periods=n_bars, freq='B')
    
    trend = np.linspace(0, 50, n_bars)
    noise = np.cumsum(np.random.randn(n_bars) * 0.8)
    cycles = 15 * np.sin(np.linspace(0, 12 * np.pi, n_bars))
    momentum = np.cumsum(np.random.randn(n_bars) * 0.3)
    
    base_price = 400 + trend + noise + cycles + momentum
    base_price = np.maximum(base_price, 300)
    
    daily_range = np.abs(np.random.randn(n_bars) * 2) + 1
    
    return pd.DataFrame({
        'open': base_price + np.random.randn(n_bars) * 0.5,
        'high': base_price + daily_range,
        'low': base_price - daily_range,
        'close': base_price,
        'volume': np.random.randint(50000000, 150000000, n_bars)
    }, index=dates)


def split_data(df: pd.DataFrame, train_ratio: float = 0.6) -> tuple:
    """Split data into training and test sets."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def prepare_training_data(train_df: pd.DataFrame, tda_gen: TDAFeatureGenerator,
                          preprocessor: DataPreprocessor) -> tuple:
    """Generate TDA features and prepare sequences for training."""
    tda_features = tda_gen.generate_features(train_df)
    X, y = preprocessor.prepare_sequences(train_df, tda_features)
    return X.astype(np.float32), y.astype(np.float32)


def run_backtest(test_df: pd.DataFrame, model: NeuralNetPredictor,
                 tda_gen: TDAFeatureGenerator, preprocessor: DataPreprocessor,
                 initial_cash: float = 100000) -> dict:
    """Run backtest with trained model and return performance metrics."""
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
    
    return extract_metrics(results, initial_cash, cerebro.broker.getvalue())


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


def main(data_path: str = None, results_path: str = None):
    """Main execution: load data, train model, run backtest, save results."""
    if results_path is None:
        results_path = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/backtest_results.json'
    
    print("Loading data...")
    df = load_data(data_path)
    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    print("Splitting train/test data...")
    train_df, test_df = split_data(df, train_ratio=0.7)
    print(f"  Train: {len(train_df)} bars, Test: {len(test_df)} bars")
    
    print("Generating TDA features...")
    tda_gen = TDAFeatureGenerator(window=20, embedding_dim=3)
    preprocessor = DataPreprocessor(sequence_length=15)
    
    X_train, y_train = prepare_training_data(train_df, tda_gen, preprocessor)
    print(f"  Training samples: {len(X_train)}")
    
    print("Training neural network...")
    model = NeuralNetPredictor(sequence_length=15, n_features=6, lstm_units=64)
    model.compile_model(learning_rate=0.0005)
    
    train_model(model, X_train, y_train, epochs=100, batch_size=16)
    print("  Training complete")
    
    weights_path = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/model_weights.weights.h5'
    model.save_checkpoint(weights_path)
    print(f"  Model saved to {weights_path}")
    
    print("Running backtest...")
    metrics = run_backtest(test_df, model, tda_gen, preprocessor)
    
    save_results(metrics, results_path)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return:   {metrics['total_return']*100:.2f}%")
    print(f"  Max Drawdown:   {metrics['max_drawdown']*100:.2f}%")
    print(f"  Num Trades:     {metrics['num_trades']}")
    print(f"  Win Rate:       {metrics['win_rate']*100:.2f}%")
    print(f"  Avg Win/Loss:   {metrics['avg_win_loss']:.2f}")
    print(f"  Final Value:    ${metrics['final_value']:,.2f}")
    print("=" * 50)
    
    return metrics


if __name__ == "__main__":
    data_file = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/test_data/spy_2022_2024.csv'
    if not os.path.exists(data_file):
        data_file = None
    
    metrics = main(data_path=data_file)
