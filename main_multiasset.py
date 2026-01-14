"""Multi-Asset TDA+NN Portfolio Strategy.

Engine V1.3: Feature & TDA Enrichment (top_k lifetimes, count_large, wasserstein).

Features:
- V1.3 NEW: Enriched TDA features (20 total: top_k, count_large, wasserstein_approx)
- V1.3 NEW: Regime labeling based on returns, volatility, TDA entropy
- V1.3 NEW: Feature diagnostics & ablation experiments (v1.1/v1.2/v1.3)
- V1.2-data: Polygon (Massive/OTREP) as primary data provider, yfinance fallback
- V1.2: Walk-forward validation, richer TDA, expanded universe
- V1.1: Cost-aware, risk-aware multi-asset tactical allocator

Tickers: SPY, QQQ, IWM, XLF, XLK (core) + expanded set
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtrader as bt

from src.tda_features import TDAFeatureGenerator
from src.nn_predictor import NeuralNetPredictor, DataPreprocessor, train_model
from src.ensemble_strategy import EnsembleStrategy, PerformanceAnalyzer
from src.data.data_provider import get_ohlcv_data, validate_provider
from src.regime_labeler import RegimeLabeler, label_regimes
from src.risk_management import RiskManager, TradeJournal, calculate_atr
from src.transaction_costs import CostModel, CostScenario, estimate_roundtrip_cost

# Set random seeds for reproducibility
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIGURATION - ENGINE V1.3
# =============================================================================

# Run mode: "baseline", "robustness", "walkforward", "expanded_universe", "ablation"
MODE = "baseline"

# =============================================================================
# DATA PROVIDER CONFIGURATION (V1.2-data)
# =============================================================================

# Primary data provider: "polygon" (Massive/OTREP) or "yfinance" (fallback)
DATA_PROVIDER = "yfinance"  # Change to "polygon" when POLYGON_API_KEY_OTREP is set

# Environment variable containing the Polygon/Massive OTREP API key
POLYGON_API_KEY_ENV = "POLYGON_API_KEY_OTREP"

# Timeframe for OHLCV data
# NOTE: In future (V2.x), switching DEFAULT_TIMEFRAME to e.g. "60m"
# and adjusting Backtrader's timeframe/compression will enable intraday tests.
DEFAULT_TIMEFRAME = "1d"

# Multi-asset configuration (core universe)
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK"]

# Expanded universe (V1.2)
EXPANDED_TICKERS = [
    # Core ETFs (original)
    "SPY", "QQQ", "IWM", "XLF", "XLK",
    # Additional sector ETFs
    "XLV", "XLY", "XLP", "XLI", "XLE",
    # Large-cap single names
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "UNH", "XOM"
]

# Date ranges (used for baseline mode)
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"  # V1.2: Use recommended 2-year training
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Walk-forward configuration (V1.2)
WALK_FORWARD_CONFIG = {
    "start": "2020-01-01",
    "end": "2025-12-31",
    "train_window_years": 2,
    "test_window_months": 6,
    "step_months": 6,  # Non-overlapping test periods
}

# Model parameters
SEQUENCE_LENGTH = 15
LSTM_UNITS = 64
LEARNING_RATE = 0.0005
EPOCHS = 100
BATCH_SIZE = 16

# =============================================================================
# V1.3: TDA FEATURE MODE & ABLATION CONFIGURATION
# =============================================================================
# Feature modes: 'v1.1' (4 TDA), 'v1.2' (10 TDA), 'v1.3' (20 TDA)
TDA_FEATURE_MODE = "v1.3"  # V1.3 enriched features (default)

# N_FEATURES computed dynamically based on TDA_FEATURE_MODE:
#   v1.1: 4 TDA + 2 OHLCV = 6 total
#   v1.2: 10 TDA + 2 OHLCV = 12 total  
#   v1.3: 20 TDA + 2 OHLCV = 22 total
TDA_FEATURE_COUNTS = {'v1.1': 4, 'v1.2': 10, 'v1.3': 20}
N_FEATURES = TDA_FEATURE_COUNTS[TDA_FEATURE_MODE] + 2  # TDA + OHLCV-derived

# V1.2 backward compat (deprecated - use TDA_FEATURE_MODE instead)
USE_EXTENDED_TDA = TDA_FEATURE_MODE in ('v1.2', 'v1.3')

# Strategy parameters
NN_BUY_THRESHOLD = 0.52
NN_SELL_THRESHOLD = 0.48
POSITION_SIZE_PCT = 0.20
USE_CONFIDENCE_SIZING = True
MIN_POSITION_PCT = 0.05
MAX_POSITION_PCT = 0.25

# Cost model parameters (V1.1+)
COST_BP_PER_SIDE = 5  # 0.05% per side (slippage + market impact estimate)

# V4.0: Risk Management Configuration - OPTIMIZED for 60%+ win rate strategies
USE_RISK_MANAGEMENT = True  # Enable risk management framework
RISK_PER_TRADE = 0.02       # 2% risk per trade (up from 1%, justified by strong win rates)
STOP_ATR_MULTIPLIER = 2.0   # ATR multiplier for stop-losses
RISK_REWARD_RATIO = 2.0     # Take-profit R:R ratio
MAX_PORTFOLIO_HEAT = 0.35   # Maximum 35% portfolio heat (up from 20%, allows 2-3 concurrent positions)

# V4.0: Transaction Cost Configuration
USE_ENHANCED_COST_MODEL = True  # Enable detailed transaction cost tracking
COST_SCENARIO = 'baseline'      # 'low_cost', 'baseline', 'high_cost', 'extreme'

# Paths
RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'
WEIGHTS_PATH = f'{RESULTS_DIR}/multiasset_weights.weights.h5'
DIAGNOSTICS_PATH = f'{RESULTS_DIR}/multiasset_nn_diagnostics.txt'
RESULTS_JSON_PATH = f'{RESULTS_DIR}/multiasset_backtest.json'
ROBUSTNESS_JSON_PATH = f'{RESULTS_DIR}/multiasset_robustness_report.json'
WALKFORWARD_JSON_PATH = f'{RESULTS_DIR}/multiasset_walkforward_report.json'
EXPANDED_JSON_PATH = f'{RESULTS_DIR}/expanded_universe_backtest.json'

# =============================================================================
# ROBUSTNESS ANALYSIS SCENARIOS
# =============================================================================

SCENARIOS = [
    {
        "name": "train_2022_test_2023_2025",
        "train_start": "2022-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2023-01-01",
        "test_end":    "2025-12-31",
    },
    {
        "name": "train_2022_2023_test_2024_2025",
        "train_start": "2022-01-01",
        "train_end":   "2023-12-31",
        "test_start":  "2024-01-01",
        "test_end":    "2025-12-31",
    },
    {
        "name": "train_2022_test_2023_only",
        "train_start": "2022-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2023-01-01",
        "test_end":    "2023-12-31",
    },
    {
        "name": "train_2022_test_2024_only",
        "train_start": "2022-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2024-01-01",
        "test_end":    "2024-12-31",
    },
    {
        "name": "train_2022_test_2025_only",
        "train_start": "2022-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2025-01-01",
        "test_end":    "2025-12-31",
    },
]


# =============================================================================
# PERFORMANCE-WEIGHTED PORTFOLIO HELPER
# =============================================================================

def compute_performance_weights(per_asset_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute simple performance-based portfolio weights from per-asset metrics.

    Design:
      - score_i = max(Sharpe_i, 0)
      - Exclude (give zero weight to) assets that are both:
          * num_trades < 3 OR Sharpe_i < -0.3, AND
          * total_return <= 0
        (e.g., QQQ with negative Sharpe and return, SPY with 0 trades and 0 return)
      - For remaining assets, w_i ∝ score_i.
      - Clamp each weight to [0, 0.7] to prevent a single asset dominating.
      - Renormalize to sum to 1.0.
      - If all scores are zero (edge case), fall back to equal weights across all assets.
    """
    scores = {}
    
    for ticker, m in per_asset_results.items():
        sharpe = m.get("sharpe_ratio", 0.0)
        total_return = m.get("total_return", 0.0)
        num_trades = m.get("num_trades", 0)

        # Exclusion rule: exclude assets with poor trading AND non-positive return
        if ((num_trades < 3 or sharpe < -0.3) and total_return <= 0):
            scores[ticker] = 0.0
        else:
            scores[ticker] = max(sharpe, 0.0)
    
    total_score = sum(scores.values())
    
    # Edge case: all scores are zero -> fall back to equal weights
    if total_score == 0:
        n_assets = len(per_asset_results)
        return {ticker: 1.0 / n_assets for ticker in per_asset_results.keys()}
    
    # Normalize to preliminary weights
    weights = {ticker: score / total_score for ticker, score in scores.items()}
    
    # Clamp each weight to [0.0, 0.7]
    MAX_WEIGHT = 0.7
    clamped = False
    for ticker in weights:
        if weights[ticker] > MAX_WEIGHT:
            weights[ticker] = MAX_WEIGHT
            clamped = True
    
    # Renormalize after clamping
    if clamped:
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {ticker: w / total_weight for ticker, w in weights.items()}
    
    return weights


# =============================================================================
# PHASE A: MULTI-ASSET DATA & FEATURE PIPELINE
# =============================================================================

def download_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from the configured data provider.
    
    V1.2-data: Uses unified data layer (Polygon primary, yfinance fallback).
    """
    print(f"    Downloading {ticker} via {DATA_PROVIDER} ({DEFAULT_TIMEFRAME})...")
    try:
        df = get_ohlcv_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe=DEFAULT_TIMEFRAME,
            provider=DATA_PROVIDER,
            polygon_api_key_env=POLYGON_API_KEY_ENV,
        )
        
        if df.empty:
            print(f"    ⚠️  No data for {ticker}")
            return pd.DataFrame()
        
        # Ensure consistent columns and index (data layer should already do this)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        
        print(f"    ✓ {ticker}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"    ✗ Error downloading {ticker}: {e}")
        return pd.DataFrame()


def generate_tda_features_and_labels(df: pd.DataFrame, 
                                      tda_gen: TDAFeatureGenerator,
                                      preprocessor: DataPreprocessor) -> Tuple[np.ndarray, np.ndarray]:
    """Generate TDA features and binary labels for a price DataFrame.
    
    Labels: 1 if next close > current close, else 0
    """
    if len(df) < 50:  # Need enough data for TDA warmup
        return np.array([]), np.array([])
    
    tda_features = tda_gen.generate_features(df)
    X, y = preprocessor.prepare_sequences(df, tda_features)
    
    return X.astype(np.float32), y.astype(np.float32)


def build_multiasset_dataset(tickers: List[str], 
                              train_start: str, train_end: str,
                              test_start: str, test_end: str,
                              tda_gen: TDAFeatureGenerator,
                              preprocessor: DataPreprocessor) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build multi-asset dataset for training and testing.
    
    Returns:
        train_X: Concatenated training features across all tickers (N_total, seq_len, n_features)
        train_y: Concatenated training labels (N_total,)
        test_data_per_ticker: Dict[ticker] = {
            'df': test DataFrame,
            'X': test features,
            'y': test labels
        }
    """
    print("\n" + "=" * 60)
    print("PHASE A: Building Multi-Asset Dataset")
    print("=" * 60)
    
    train_X_list = []
    train_y_list = []
    test_data_per_ticker = {}
    
    for ticker in tickers:
        print(f"\n  [{ticker}]")
        
        # Download full date range
        df_full = download_ticker_data(ticker, train_start, test_end)
        if df_full.empty:
            continue
        
        # Split into train and test
        train_df = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)].copy()
        test_df = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)].copy()
        
        print(f"    Train: {len(train_df)} bars, Test: {len(test_df)} bars")
        
        # Generate training features and labels
        if len(train_df) >= 50:
            X_train, y_train = generate_tda_features_and_labels(train_df, tda_gen, preprocessor)
            if len(X_train) > 0:
                train_X_list.append(X_train)
                train_y_list.append(y_train)
                print(f"    Train samples: {len(X_train)}")
        
        # Generate test features and labels
        if len(test_df) >= 50:
            X_test, y_test = generate_tda_features_and_labels(test_df, tda_gen, preprocessor)
            test_data_per_ticker[ticker] = {
                'df': test_df,
                'X': X_test,
                'y': y_test
            }
            print(f"    Test samples: {len(X_test)}")
    
    # Concatenate training data
    if train_X_list:
        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
    else:
        train_X = np.array([])
        train_y = np.array([])
    
    print(f"\n  Combined training data:")
    print(f"    train_X shape: {train_X.shape}")
    print(f"    train_y shape: {train_y.shape}")
    print(f"    Class balance: {np.mean(train_y)*100:.1f}% up, {(1-np.mean(train_y))*100:.1f}% down")
    print(f"    Tickers with test data: {list(test_data_per_ticker.keys())}")
    
    return train_X, train_y, test_data_per_ticker


# =============================================================================
# PHASE B: MULTI-ASSET NN TRAINING
# =============================================================================

def train_multiasset_nn(train_X: np.ndarray, train_y: np.ndarray,
                         diagnostics_path: str) -> NeuralNetPredictor:
    """
    Train single NN on pooled multi-asset data.
    
    Logs diagnostics to file and stdout.
    """
    print("\n" + "=" * 60)
    print("PHASE B1: Training Multi-Asset Neural Network")
    print("=" * 60)
    
    # Create model
    model = NeuralNetPredictor(
        sequence_length=SEQUENCE_LENGTH, 
        n_features=N_FEATURES, 
        lstm_units=LSTM_UNITS
    )
    
    # Compile with entropy penalty
    model.compile_model(
        learning_rate=LEARNING_RATE, 
        use_entropy_penalty=True, 
        entropy_weight=0.05
    )
    
    # Build model
    _ = model(train_X[:1])
    
    # Train
    print(f"\n  Training on {len(train_X)} samples...")
    history, spread_callback = train_model(
        model, train_X, train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        track_output_spread=True,
        verbose_spread=True
    )
    
    # Diagnostics
    predictions = model.predict(train_X, verbose=0).flatten()
    
    diagnostics = []
    diagnostics.append(f"Multi-Asset NN Training Diagnostics - {datetime.now().isoformat()}")
    diagnostics.append("=" * 60)
    diagnostics.append(f"\nTraining Data:")
    diagnostics.append(f"  train_X shape: {train_X.shape}")
    diagnostics.append(f"  train_y shape: {train_y.shape}")
    diagnostics.append(f"  train_y distribution: {np.mean(train_y)*100:.1f}% up")
    diagnostics.append(f"\nTraining Results:")
    diagnostics.append(f"  Epochs completed: {len(history.history.get('loss', []))}")
    diagnostics.append(f"  Final loss: {history.history['loss'][-1]:.4f}")
    diagnostics.append(f"  Final val_loss: {history.history.get('val_loss', [0])[-1]:.4f}")
    diagnostics.append(f"\nPrediction Statistics on Training Data:")
    diagnostics.append(f"  Mean: {np.mean(predictions):.4f}")
    diagnostics.append(f"  Std: {np.std(predictions):.4f}")
    diagnostics.append(f"  % > 0.55: {np.mean(predictions > 0.55)*100:.1f}%")
    diagnostics.append(f"  % < 0.45: {np.mean(predictions < 0.45)*100:.1f}%")
    diagnostics.append(f"  % in extreme zones: {np.mean((predictions > 0.55) | (predictions < 0.45))*100:.1f}%")
    
    # Print and save
    for line in diagnostics:
        print(f"  {line}")
    
    os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
    with open(diagnostics_path, 'w') as f:
        f.write('\n'.join(diagnostics))
    
    print(f"\n  Diagnostics saved to: {diagnostics_path}")
    
    # Save weights
    model.save_checkpoint(WEIGHTS_PATH)
    print(f"  Model saved to: {WEIGHTS_PATH}")
    
    return model


# =============================================================================
# PHASE B: MULTI-ASSET BACKTESTING
# =============================================================================

def run_single_asset_backtest(ticker: str, 
                               test_df: pd.DataFrame,
                               model: NeuralNetPredictor,
                               tda_gen: TDAFeatureGenerator,
                               preprocessor: DataPreprocessor,
                               initial_cash: float = 20000,  # Per-asset allocation
                               enable_diagnostics: bool = False,
                               risk_manager: RiskManager = None,
                               trade_journal: TradeJournal = None) -> Dict[str, Any]:
    """
    Run backtest for a single asset.
    
    V4.0: Now includes risk management metrics.
    
    Returns metrics dictionary.
    """
    print(f"\n  Running backtest for {ticker} ({len(test_df)} bars)...")
    
    cerebro = bt.Cerebro()
    
    # Add data
    data = bt.feeds.PandasData(dataname=test_df)
    cerebro.adddata(data)
    
    # Add strategy with ticker-specific CSV path
    csv_path = f'{RESULTS_DIR}/diagnostics_{ticker}.csv' if enable_diagnostics else None
    
    # V4.0: Create risk manager for this asset if not provided
    if risk_manager is None and USE_RISK_MANAGEMENT:
        risk_manager = RiskManager(
            initial_capital=initial_cash,
            risk_per_trade=RISK_PER_TRADE,
            log_path=f'{RESULTS_DIR}/risk_log_{ticker}.csv'
        )
    
    # V4.0: Create trade journal for this asset if not provided
    if trade_journal is None and USE_RISK_MANAGEMENT:
        trade_journal = TradeJournal(
            journal_path=f'{RESULTS_DIR}/trade_journal_{ticker}.csv'
        )
    
    cerebro.addstrategy(
        EnsembleStrategy,
        nn_model=model,
        tda_generator=tda_gen,
        preprocessor=preprocessor,
        sequence_length=SEQUENCE_LENGTH,
        ticker=ticker,
        nn_buy_threshold=NN_BUY_THRESHOLD,
        nn_sell_threshold=NN_SELL_THRESHOLD,
        use_confidence_sizing=USE_CONFIDENCE_SIZING,
        min_position_pct=MIN_POSITION_PCT,
        max_position_pct=MAX_POSITION_PCT,
        # V4.0: Risk management parameters
        use_risk_management=USE_RISK_MANAGEMENT,
        initial_capital=initial_cash,
        risk_per_trade=RISK_PER_TRADE,
        stop_atr_multiplier=STOP_ATR_MULTIPLIER,
        risk_reward_ratio=RISK_REWARD_RATIO,
        max_portfolio_heat=MAX_PORTFOLIO_HEAT,
        risk_manager=risk_manager,
        trade_journal=trade_journal,
        verbose=False,
        diagnostic_csv_path=csv_path if csv_path else '',
        enable_diagnostics=enable_diagnostics,
    )
    
    # Analyzers
    cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    # Broker settings
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)
    
    # Run
    results = cerebro.run()
    final_value_gross = cerebro.broker.getvalue()
    
    # Extract metrics
    strat = results[0]
    
    # Daily returns for Sharpe
    returns_dict = strat.analyzers.returns.get_analysis()
    daily_returns = list(returns_dict.values()) if returns_dict else []
    
    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe_gross = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe_gross = 0
    
    # Drawdown
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.get('max', {}).get('drawdown', 0) or 0
    
    # Performance (including turnover from V1.1 PerformanceAnalyzer)
    perf = strat.analyzers.performance.get_analysis()
    
    num_trades = int(perf.get('num_trades', 0))
    win_rate = float(perf.get('win_rate', 0))
    total_notional_traded = float(perf.get('total_notional_traded', 0.0))
    turnover = float(perf.get('turnover', 0.0))
    
    # V1.1: Cost-aware metrics
    # Extra costs = slippage/impact estimate on top of commission (already deducted by broker)
    extra_costs = total_notional_traded * (COST_BP_PER_SIDE / 10000.0)
    final_value_net = final_value_gross - extra_costs
    
    total_return_gross = (final_value_gross - initial_cash) / initial_cash
    total_return_net = (final_value_net - initial_cash) / initial_cash
    
    # Approximate Sharpe_net by scaling (exact would require cost-adjusted daily returns)
    if total_return_gross != 0:
        sharpe_net = sharpe_gross * (total_return_net / total_return_gross)
    else:
        sharpe_net = sharpe_gross
    
    metrics = {
        'ticker': ticker,
        # Gross metrics
        'sharpe_ratio': round(float(sharpe_gross), 4),
        'total_return': round(float(total_return_gross), 4),
        'max_drawdown': round(float(max_dd) / 100, 4),
        'num_trades': num_trades,
        'win_rate': round(win_rate, 4),
        'final_value': round(float(final_value_gross), 2),
        'initial_cash': initial_cash,
        # V1.1: Cost-aware metrics
        'total_notional_traded': round(total_notional_traded, 2),
        'turnover': round(turnover, 4),
        'extra_costs': round(extra_costs, 2),
        'sharpe_ratio_net': round(float(sharpe_net), 4),
        'total_return_net': round(float(total_return_net), 4),
        'final_value_net': round(float(final_value_net), 2),
        # Daily returns for portfolio aggregation
        'daily_returns': daily_returns
    }
    
    # V4.0: Add risk management metrics
    if USE_RISK_MANAGEMENT and hasattr(strat, 'get_risk_metrics'):
        risk_metrics = strat.get_risk_metrics()
        metrics['risk_management'] = {
            'num_stopped_out': risk_metrics.get('num_stopped_out', 0),
            'num_take_profit_hits': risk_metrics.get('num_take_profit_hits', 0),
            'max_portfolio_heat_reached': risk_metrics.get('max_portfolio_heat_reached', 0),
            'avg_r_multiple': risk_metrics.get('avg_r_multiple', 0),
            'profit_factor': risk_metrics.get('profit_factor', 0),
            'expectancy_per_trade': risk_metrics.get('expectancy_per_trade', 0),
        }
        print(f"      Risk: stops={risk_metrics.get('num_stopped_out', 0)}, "
              f"targets={risk_metrics.get('num_take_profit_hits', 0)}, "
              f"heat={risk_metrics.get('max_portfolio_heat_reached', 0)*100:.1f}%")
    
    print(f"    {ticker}: Sharpe={sharpe_gross:.2f} (net:{sharpe_net:.2f}), Return={total_return_gross*100:.2f}% (net:{total_return_net*100:.2f}%), Trades={num_trades}")
    
    return metrics


def run_multiasset_backtest(test_data_per_ticker: Dict,
                             model: NeuralNetPredictor,
                             tda_gen: TDAFeatureGenerator,
                             preprocessor: DataPreprocessor,
                             total_cash: float = 100000) -> Dict[str, Any]:
    """
    Run backtests for all assets and aggregate results.
    
    Returns per-asset and portfolio-level metrics.
    """
    print("\n" + "=" * 60)
    print("PHASE B2: Running Multi-Asset Backtests")
    print("=" * 60)
    
    n_assets = len(test_data_per_ticker)
    per_asset_cash = total_cash / n_assets if n_assets > 0 else total_cash
    
    print(f"\n  Total capital: ${total_cash:,.0f}")
    print(f"  Per-asset allocation: ${per_asset_cash:,.0f}")
    print(f"  Assets: {list(test_data_per_ticker.keys())}")
    
    per_asset_results = {}
    all_daily_returns = []
    
    for ticker, data in test_data_per_ticker.items():
        test_df = data['df']
        
        metrics = run_single_asset_backtest(
            ticker=ticker,
            test_df=test_df,
            model=model,
            tda_gen=tda_gen,
            preprocessor=preprocessor,
            initial_cash=per_asset_cash,
            enable_diagnostics=(ticker == list(test_data_per_ticker.keys())[0])  # Only first asset
        )
        
        per_asset_results[ticker] = {k: v for k, v in metrics.items() if k != 'daily_returns'}
        
        # Collect daily returns for portfolio calculation
        if metrics['daily_returns']:
            all_daily_returns.append(metrics['daily_returns'])
    
    return per_asset_results, all_daily_returns


def compute_portfolio_metrics(per_asset_results: Dict[str, Any],
                               all_daily_returns: List[List[float]],
                               total_cash: float = 100000,
                               weights: Dict[str, float] = None,
                               label: str = "Equal-Weight") -> Dict[str, Any]:
    """
    Compute portfolio metrics with optional custom weights.
    
    V1.1: Now includes both gross and net (cost-adjusted) metrics.
    
    Args:
        per_asset_results: Per-asset backtest results
        all_daily_returns: Daily returns per asset (list of lists)
        total_cash: Total portfolio capital
        weights: Optional dict of ticker -> weight. If None, uses equal weights.
        label: Label for printing (e.g., "Equal-Weight" or "Performance-Weighted")
    """
    print(f"\n  Computing {label} Portfolio Metrics...")
    
    n_assets = len(per_asset_results)
    asset_keys = list(per_asset_results.keys())
    
    if n_assets == 0:
        return {
            'sharpe_ratio': 0, 'sharpe_ratio_net': 0,
            'total_return': 0, 'total_return_net': 0,
            'max_drawdown': 0, 'num_trades': 0, 'win_rate': 0,
            'turnover': 0,
            'final_value': total_cash, 'final_value_net': total_cash,
            'initial_cash': total_cash
        }
    
    # Determine weights
    if weights is None:
        weights_used = {ticker: 1.0 / n_assets for ticker in asset_keys}
    else:
        weights_used = {ticker: weights.get(ticker, 0.0) for ticker in asset_keys}
        total_weight = sum(weights_used.values())
        if total_weight > 0:
            weights_used = {t: w / total_weight for t, w in weights_used.items()}
        else:
            weights_used = {ticker: 1.0 / n_assets for ticker in asset_keys}
    
    # Aggregate trade metrics
    total_trades = sum(m['num_trades'] for m in per_asset_results.values())
    total_wins = sum(m['num_trades'] * m['win_rate'] for m in per_asset_results.values())
    overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
    
    # V1.1: Weighted capital allocation with gross and net final values
    portfolio_final_gross = 0.0
    portfolio_final_net = 0.0
    portfolio_turnover = 0.0
    
    for ticker in asset_keys:
        w_i = weights_used[ticker]
        capital_i = total_cash * w_i
        initial_cash_i = per_asset_results[ticker]["initial_cash"]
        final_gross_i = per_asset_results[ticker]["final_value"]
        extra_costs_i = per_asset_results[ticker].get("extra_costs", 0.0)
        turnover_i = per_asset_results[ticker].get("turnover", 0.0)
        
        if initial_cash_i > 0:
            scale_factor = capital_i / initial_cash_i
            scaled_final_gross_i = final_gross_i * scale_factor
            scaled_extra_costs_i = extra_costs_i * scale_factor
            scaled_final_net_i = scaled_final_gross_i - scaled_extra_costs_i
        else:
            scaled_final_gross_i = capital_i
            scaled_final_net_i = capital_i
        
        portfolio_final_gross += scaled_final_gross_i
        portfolio_final_net += scaled_final_net_i
        portfolio_turnover += w_i * turnover_i
    
    total_return_gross = (portfolio_final_gross - total_cash) / total_cash if total_cash > 0 else 0
    total_return_net = (portfolio_final_net - total_cash) / total_cash if total_cash > 0 else 0
    
    # Max drawdown (conservative: worst across assets)
    max_dd = max(m['max_drawdown'] for m in per_asset_results.values())
    
    # Portfolio Sharpe from weighted daily returns
    if all_daily_returns and len(all_daily_returns) == n_assets:
        max_len = max(len(r) for r in all_daily_returns)
        padded_returns = []
        for returns in all_daily_returns:
            padded = returns + [0] * (max_len - len(returns))
            padded_returns.append(padded)
        
        asset_weights_array = np.array([weights_used[ticker] for ticker in asset_keys], dtype=float)
        if asset_weights_array.sum() > 0:
            asset_weights_norm = asset_weights_array / asset_weights_array.sum()
        else:
            asset_weights_norm = np.ones_like(asset_weights_array) / max(len(asset_weights_array), 1)
        
        padded_returns_array = np.array(padded_returns)
        portfolio_returns = np.average(padded_returns_array, axis=0, weights=asset_weights_norm)
        
        if len(portfolio_returns) > 1:
            mean_ret = np.mean(portfolio_returns)
            std_ret = np.std(portfolio_returns)
            sharpe_gross = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe_gross = 0
    else:
        sharpe_gross = 0
    
    # V1.1: Approximate Sharpe_net by scaling (exact would require cost-adjusted daily returns)
    if total_return_gross != 0:
        sharpe_net = sharpe_gross * (total_return_net / total_return_gross)
    else:
        sharpe_net = sharpe_gross
    
    portfolio_metrics = {
        'sharpe_ratio': round(float(sharpe_gross), 4),
        'sharpe_ratio_net': round(float(sharpe_net), 4),
        'total_return': round(float(total_return_gross), 4),
        'total_return_net': round(float(total_return_net), 4),
        'max_drawdown': round(float(max_dd), 4),
        'num_trades': total_trades,
        'win_rate': round(float(overall_win_rate), 4),
        'turnover': round(float(portfolio_turnover), 4),
        'final_value': round(float(portfolio_final_gross), 2),
        'final_value_net': round(float(portfolio_final_net), 2),
        'initial_cash': total_cash,
        'n_assets': n_assets
    }
    
    print(f"    Sharpe: {sharpe_gross:.4f} (net: {sharpe_net:.4f})")
    print(f"    Return: {total_return_gross*100:.2f}% (net: {total_return_net*100:.2f}%)")
    print(f"    Max DD: {max_dd*100:.2f}%")
    print(f"    Trades: {total_trades}, Turnover: {portfolio_turnover:.2f}x")
    print(f"    Final Value: ${portfolio_final_gross:,.2f} (net: ${portfolio_final_net:,.2f})")
    
    return portfolio_metrics


# =============================================================================
# V1.1: RISK OVERLAY
# =============================================================================

def compute_risk_scale_from_portfolio_eq(eq_metrics: Dict[str, Any]) -> float:
    """
    Compute a coarse risk scaling factor in [0.5, 1.0] based on equal-weight
    portfolio performance for the scenario.

    Rules:
      - If eq_metrics["sharpe_ratio"] < 0 or eq_metrics["total_return"] <= 0:
          return 0.5  # risk-off
      - Elif eq_metrics["sharpe_ratio"] < 0.5:
          return 0.75  # moderate risk
      - Else:
          return 1.0  # risk-on

    This is a placeholder for a true TDA-based turbulence/regime model.
    """
    sharpe = eq_metrics.get("sharpe_ratio", 0.0)
    total_return = eq_metrics.get("total_return", 0.0)
    
    if sharpe < 0 or total_return <= 0:
        return 0.5  # risk-off
    elif sharpe < 0.5:
        return 0.75  # moderate risk
    else:
        return 1.0  # risk-on


# =============================================================================
# PHASE C: THRESHOLD SENSITIVITY ANALYSIS
# =============================================================================

def threshold_sensitivity_analysis(test_data_per_ticker: Dict,
                                    model: NeuralNetPredictor,
                                    tda_gen: TDAFeatureGenerator,
                                    preprocessor: DataPreprocessor) -> Dict:
    """
    Analyze trade counts at different threshold levels.
    
    Computes metrics for:
    - Baseline: 0.52/0.48
    - Conservative: 0.53/0.47
    - Aggressive: 0.51/0.49
    
    Returns analysis summary (does NOT change live thresholds).
    """
    print("\n" + "=" * 60)
    print("PHASE C: Threshold Sensitivity Analysis")
    print("=" * 60)
    
    thresholds = [
        ('conservative', 0.53, 0.47),
        ('baseline', 0.52, 0.48),
        ('aggressive', 0.51, 0.49),
    ]
    
    analysis = {}
    
    # For efficiency, just analyze prediction distribution rather than full backtest
    ticker = list(test_data_per_ticker.keys())[0]  # Use first ticker as sample
    test_df = test_data_per_ticker[ticker]['df']
    
    # Get predictions for sample
    print(f"\n  Analyzing on {ticker} test data...")
    
    # Build prediction array from test data
    # (simplified: just count potential signals based on thresholds)
    
    # Get NN predictions for test data
    X_test = test_data_per_ticker[ticker]['X']
    if len(X_test) > 0:
        predictions = model.predict(X_test, verbose=0).flatten()
        
        print(f"  Prediction distribution:")
        print(f"    Mean: {np.mean(predictions):.4f}")
        print(f"    Std: {np.std(predictions):.4f}")
        print(f"    Min: {np.min(predictions):.4f}")
        print(f"    Max: {np.max(predictions):.4f}")
        
        print(f"\n  Signal counts by threshold:")
        for name, buy_thresh, sell_thresh in thresholds:
            buy_signals = np.sum(predictions > buy_thresh)
            sell_signals = np.sum(predictions < sell_thresh)
            neutral = len(predictions) - buy_signals - sell_signals
            
            analysis[name] = {
                'buy_threshold': buy_thresh,
                'sell_threshold': sell_thresh,
                'buy_signals': int(buy_signals),
                'sell_signals': int(sell_signals),
                'neutral': int(neutral),
                'pct_buy': round(buy_signals / len(predictions) * 100, 1),
                'pct_sell': round(sell_signals / len(predictions) * 100, 1),
            }
            
            print(f"    {name:12s} ({buy_thresh}/{sell_thresh}): "
                  f"BUY={buy_signals} ({analysis[name]['pct_buy']}%), "
                  f"SELL={sell_signals} ({analysis[name]['pct_sell']}%)")
    
    # Recommendation
    print(f"\n  Recommendation:")
    if analysis.get('aggressive', {}).get('pct_buy', 0) > 30:
        print("    → Current thresholds (0.52/0.48) appear reasonable.")
        print("    → Aggressive thresholds (0.51/0.49) would generate more signals.")
    else:
        print("    → Model has bearish bias. Consider adjusting training data or features.")
    
    return analysis


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_multiasset_results(per_asset_results: Dict,
                             portfolio_equal_weight: Dict,
                             portfolio_weighted: Dict,
                             weights_by_ticker: Dict[str, float],
                             threshold_analysis: Dict,
                             output_path: str):
    """Save all results to JSON file including both portfolio types."""
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'tickers': list(per_asset_results.keys()),
            'train_period': f"{TRAIN_START} to {TRAIN_END}",
            'test_period': f"{TEST_START} to {TEST_END}",
            'thresholds': {
                'nn_buy_threshold': NN_BUY_THRESHOLD,
                'nn_sell_threshold': NN_SELL_THRESHOLD,
            },
            'confidence_sizing': USE_CONFIDENCE_SIZING,
        },
        'per_asset': per_asset_results,
        'portfolio_equal_weight': portfolio_equal_weight,
        'portfolio_performance_weighted': portfolio_weighted,
        'weights': {ticker: round(w, 4) for ticker, w in weights_by_ticker.items()},
        'threshold_sensitivity': threshold_analysis,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {output_path}")


# =============================================================================
# SCENARIO RUNNER (for robustness analysis)
# =============================================================================

def run_scenario(
    scenario: Dict[str, str],
    tickers: List[str] = None,
    total_cash: float = 100000.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs the full multi-asset pipeline for a given train/test scenario.

    V1.1: Includes cost-aware metrics and risk overlay on performance-weighted portfolio.

    Steps:
      - Build multi-asset dataset (scenario-specific train/test).
      - Train NN on pooled train data.
      - Run multi-asset backtests.
      - Compute equal-weight and performance-weight portfolios.
      - Apply risk overlay to performance-weighted allocation.
      - Run threshold sensitivity on one ticker (as currently).
    Returns a dict with all key components for this scenario.
    """
    if tickers is None:
        tickers = TICKERS
    
    name = scenario["name"]
    train_start = scenario["train_start"]
    train_end = scenario["train_end"]
    test_start = scenario["test_start"]
    test_end = scenario["test_end"]
    
    print("\n" + "=" * 80)
    print(f"RUNNING SCENARIO: {name}")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Test:  {test_start} to {test_end}")
    print(f"  TDA Feature Mode: {TDA_FEATURE_MODE} ({TDA_FEATURE_COUNTS[TDA_FEATURE_MODE]} TDA features)")
    print(f"  Total N_FEATURES: {N_FEATURES}")
    print("=" * 80)
    
    # Initialize generators (V1.3: support enriched TDA features)
    tda_gen = TDAFeatureGenerator(window=20, embedding_dim=3, feature_mode=TDA_FEATURE_MODE)
    preprocessor = DataPreprocessor(sequence_length=SEQUENCE_LENGTH, use_extended_tda=USE_EXTENDED_TDA)
    
    # Build multi-asset dataset
    train_X, train_y, test_data_per_ticker = build_multiasset_dataset(
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        tda_gen=tda_gen,
        preprocessor=preprocessor
    )
    
    # Handle empty training data
    if len(train_X) == 0 or not test_data_per_ticker:
        print(f"  ⚠️  No data for scenario {name}. Returning zeros.")
        empty_result = {
            "name": name,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "per_asset": {},
            "portfolio_equal_weight": {"sharpe_ratio": 0, "sharpe_ratio_net": 0, "total_return": 0, "total_return_net": 0, "num_trades": 0},
            "portfolio_performance_weighted": {"sharpe_ratio": 0, "sharpe_ratio_net": 0, "total_return": 0, "total_return_net": 0, "num_trades": 0},
            "weights": {},
            "risk_scale": 0.5,
            "cash_weight": 0.5,
            "threshold_sensitivity": {},
        }
        return empty_result
    
    # Train NN
    model = train_multiasset_nn(train_X, train_y, DIAGNOSTICS_PATH)
    
    # Run multi-asset backtests
    per_asset_results, all_daily_returns = run_multiasset_backtest(
        test_data_per_ticker=test_data_per_ticker,
        model=model,
        tda_gen=tda_gen,
        preprocessor=preprocessor,
        total_cash=total_cash
    )
    
    # Compute equal-weight portfolio first
    print("\n  Computing portfolio metrics...")
    
    portfolio_equal_weight = compute_portfolio_metrics(
        per_asset_results=per_asset_results,
        all_daily_returns=all_daily_returns,
        total_cash=total_cash,
        weights=None,
        label="Equal-Weight"
    )
    
    # V1.1: Compute performance weights and apply risk overlay
    weights_by_ticker = compute_performance_weights(per_asset_results)
    risk_scale = compute_risk_scale_from_portfolio_eq(portfolio_equal_weight)
    
    # Scale weights by risk factor
    scaled_weights = {t: w * risk_scale for t, w in weights_by_ticker.items()}
    cash_weight = max(0.0, 1.0 - sum(scaled_weights.values()))
    
    print(f"\n  Risk Overlay: scale={risk_scale:.2f}, cash_weight={cash_weight*100:.1f}%")
    
    # Compute performance-weighted portfolio with risk-scaled weights
    # Note: We allocate (1 - cash_weight) * total_cash to risky assets
    # For simplicity, we still compute on full capital and note that cash portion earns 0%
    portfolio_weighted = compute_portfolio_metrics(
        per_asset_results=per_asset_results,
        all_daily_returns=all_daily_returns,
        total_cash=total_cash,
        weights=scaled_weights if sum(scaled_weights.values()) > 0 else weights_by_ticker,
        label="Performance-Weighted (Risk-Adjusted)"
    )
    
    # Add risk overlay info to portfolio
    portfolio_weighted["risk_scale"] = risk_scale
    portfolio_weighted["cash_weight"] = round(cash_weight, 4)
    
    # Threshold sensitivity
    threshold_analysis = threshold_sensitivity_analysis(
        test_data_per_ticker=test_data_per_ticker,
        model=model,
        tda_gen=tda_gen,
        preprocessor=preprocessor
    )
    
    # Print scenario summary
    print(f"\n  Scenario Summary: {name}")
    print("-" * 80)
    print(f"  {'Ticker':<8} {'Sharpe':>8} {'Sh_Net':>8} {'Return':>8} {'Ret_Net':>8} {'Trades':>7} {'Turnover':>8}")
    for ticker, m in per_asset_results.items():
        print(f"  {ticker:<8} {m['sharpe_ratio']:>8.2f} {m.get('sharpe_ratio_net',0):>8.2f} "
              f"{m['total_return']*100:>7.2f}% {m.get('total_return_net',0)*100:>7.2f}% "
              f"{m['num_trades']:>7} {m.get('turnover',0):>8.2f}x")
    print("-" * 80)
    eq = portfolio_equal_weight
    wgt = portfolio_weighted
    print(f"  {'PORTF_EQ':<8} {eq['sharpe_ratio']:>8.2f} {eq.get('sharpe_ratio_net',0):>8.2f} "
          f"{eq['total_return']*100:>7.2f}% {eq.get('total_return_net',0)*100:>7.2f}%")
    print(f"  {'PORTF_WGT':<8} {wgt['sharpe_ratio']:>8.2f} {wgt.get('sharpe_ratio_net',0):>8.2f} "
          f"{wgt['total_return']*100:>7.2f}% {wgt.get('total_return_net',0)*100:>7.2f}% "
          f"(risk_scale={risk_scale:.2f}, cash={cash_weight*100:.0f}%)")
    print(f"  Weights: {{{', '.join([f'{t}: {w*100:.0f}%' for t, w in sorted(weights_by_ticker.items(), key=lambda x: -x[1]) if w > 0.01])}}}")
    
    return {
        "name": name,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "per_asset": per_asset_results,
        "portfolio_equal_weight": portfolio_equal_weight,
        "portfolio_performance_weighted": portfolio_weighted,
        "weights": {t: round(w, 4) for t, w in weights_by_ticker.items()},
        "risk_scale": risk_scale,
        "cash_weight": round(cash_weight, 4),
        "threshold_sensitivity": threshold_analysis,
    }


# =============================================================================
# V1.2: WALK-FORWARD VALIDATION
# =============================================================================

def generate_walk_forward_folds(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate walk-forward fold schedule from configuration.
    
    Walk-forward design:
      - Rolling training window of `train_window_years`
      - Non-overlapping test periods of `test_window_months`
      - Step forward by `step_months` each fold
    
    Example with config:
      start=2020-01-01, end=2025-12-31, train=2yr, test=6mo, step=6mo
      Fold 0: train 2020-01 to 2021-12, test 2022-01 to 2022-06
      Fold 1: train 2020-07 to 2022-06, test 2022-07 to 2022-12
      ... etc.
    
    Returns list of scenario dicts compatible with run_scenario().
    """
    start = datetime.strptime(config["start"], "%Y-%m-%d")
    end = datetime.strptime(config["end"], "%Y-%m-%d")
    train_years = config["train_window_years"]
    test_months = config["test_window_months"]
    step_months = config["step_months"]
    
    folds = []
    fold_idx = 0
    
    # First test period starts after first training window
    train_start = start
    train_end = train_start + relativedelta(years=train_years) - timedelta(days=1)
    test_start = train_end + timedelta(days=1)
    test_end = test_start + relativedelta(months=test_months) - timedelta(days=1)
    
    while test_end <= end:
        fold = {
            "name": f"wf_fold_{fold_idx:02d}_{test_start.strftime('%Y%m')}",
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        }
        folds.append(fold)
        fold_idx += 1
        
        # Step forward
        train_start = train_start + relativedelta(months=step_months)
        train_end = train_start + relativedelta(years=train_years) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - timedelta(days=1)
    
    return folds


def run_walk_forward(
    tickers: List[str] = None,
    config: Dict[str, Any] = None,
    total_cash: float = 100000.0,
) -> Dict[str, Any]:
    """
    Run walk-forward validation across multiple folds.
    
    V1.2: Walk-forward provides more realistic out-of-sample evaluation by:
      - Retraining on each fold's training window
      - Testing on subsequent non-overlapping period
      - Aggregating metrics across all folds
    
    Returns:
        Dict with per-fold results and aggregated summary.
    """
    if tickers is None:
        tickers = TICKERS
    if config is None:
        config = WALK_FORWARD_CONFIG
    
    print("\n" + "=" * 80)
    print("V1.2 WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Start: {config['start']}")
    print(f"  End: {config['end']}")
    print(f"  Train Window: {config['train_window_years']} years")
    print(f"  Test Window: {config['test_window_months']} months")
    print(f"  Step: {config['step_months']} months")
    print(f"  Tickers: {tickers}")
    
    # Generate fold schedule
    folds = generate_walk_forward_folds(config)
    print(f"\n  Generated {len(folds)} folds:")
    for fold in folds:
        print(f"    {fold['name']}: train {fold['train_start']} to {fold['train_end']}, "
              f"test {fold['test_start']} to {fold['test_end']}")
    
    # Run each fold
    fold_results = []
    
    for i, fold in enumerate(folds):
        print(f"\n\n{'#'*80}")
        print(f"# WALK-FORWARD FOLD {i+1}/{len(folds)}: {fold['name']}")
        print(f"{'#'*80}")
        
        result = run_scenario(fold, tickers=tickers, total_cash=total_cash, verbose=True)
        fold_results.append(result)
    
    # Aggregate results across folds
    print("\n\n" + "=" * 100)
    print("WALK-FORWARD SUMMARY (V1.2)")
    print("=" * 100)
    
    # Collect per-fold metrics
    eq_sharpes = []
    eq_sharpes_net = []
    wgt_sharpes = []
    wgt_sharpes_net = []
    eq_returns = []
    eq_returns_net = []
    wgt_returns = []
    wgt_returns_net = []
    total_trades_all = 0
    
    print(f"\n{'Fold':<25} {'EQ_Sh':>7} {'EQ_Sh_N':>8} {'WGT_Sh':>7} {'WGT_Sh_N':>9} {'EQ_Ret%':>8} {'WGT_Ret%':>9}")
    print("-" * 100)
    
    for res in fold_results:
        eq = res["portfolio_equal_weight"]
        wgt = res["portfolio_performance_weighted"]
        
        eq_sharpes.append(eq.get("sharpe_ratio", 0))
        eq_sharpes_net.append(eq.get("sharpe_ratio_net", 0))
        wgt_sharpes.append(wgt.get("sharpe_ratio", 0))
        wgt_sharpes_net.append(wgt.get("sharpe_ratio_net", 0))
        eq_returns.append(eq.get("total_return", 0))
        eq_returns_net.append(eq.get("total_return_net", 0))
        wgt_returns.append(wgt.get("total_return", 0))
        wgt_returns_net.append(wgt.get("total_return_net", 0))
        total_trades_all += eq.get("num_trades", 0)
        
        print(f"{res['name']:<25} {eq.get('sharpe_ratio', 0):>7.2f} {eq.get('sharpe_ratio_net', 0):>8.2f} "
              f"{wgt.get('sharpe_ratio', 0):>7.2f} {wgt.get('sharpe_ratio_net', 0):>9.2f} "
              f"{eq.get('total_return', 0)*100:>7.2f}% {wgt.get('total_return', 0)*100:>8.2f}%")
    
    print("-" * 100)
    
    # Compute aggregate statistics
    n_folds = len(fold_results)
    
    aggregate = {
        # Equal-weight portfolio
        "eq_sharpe_mean": float(np.mean(eq_sharpes)) if eq_sharpes else 0,
        "eq_sharpe_std": float(np.std(eq_sharpes)) if eq_sharpes else 0,
        "eq_sharpe_net_mean": float(np.mean(eq_sharpes_net)) if eq_sharpes_net else 0,
        "eq_sharpe_net_std": float(np.std(eq_sharpes_net)) if eq_sharpes_net else 0,
        "eq_return_cumulative": float(np.sum(eq_returns)) if eq_returns else 0,
        "eq_return_net_cumulative": float(np.sum(eq_returns_net)) if eq_returns_net else 0,
        # Performance-weighted portfolio
        "wgt_sharpe_mean": float(np.mean(wgt_sharpes)) if wgt_sharpes else 0,
        "wgt_sharpe_std": float(np.std(wgt_sharpes)) if wgt_sharpes else 0,
        "wgt_sharpe_net_mean": float(np.mean(wgt_sharpes_net)) if wgt_sharpes_net else 0,
        "wgt_sharpe_net_std": float(np.std(wgt_sharpes_net)) if wgt_sharpes_net else 0,
        "wgt_return_cumulative": float(np.sum(wgt_returns)) if wgt_returns else 0,
        "wgt_return_net_cumulative": float(np.sum(wgt_returns_net)) if wgt_returns_net else 0,
        # Meta
        "n_folds": n_folds,
        "n_positive_wgt_sharpe_net": sum(1 for s in wgt_sharpes_net if s > 0),
        "n_positive_eq_sharpe_net": sum(1 for s in eq_sharpes_net if s > 0),
        "total_trades": total_trades_all,
    }
    
    print(f"\nAggregate Statistics ({n_folds} folds):")
    print(f"  EQ Portfolio:")
    print(f"    Sharpe: {aggregate['eq_sharpe_mean']:.3f} ± {aggregate['eq_sharpe_std']:.3f} (net: {aggregate['eq_sharpe_net_mean']:.3f} ± {aggregate['eq_sharpe_net_std']:.3f})")
    print(f"    Cumulative Return: {aggregate['eq_return_cumulative']*100:.2f}% (net: {aggregate['eq_return_net_cumulative']*100:.2f}%)")
    print(f"    Positive Sharpe_net in {aggregate['n_positive_eq_sharpe_net']}/{n_folds} folds")
    print(f"\n  WGT Portfolio:")
    print(f"    Sharpe: {aggregate['wgt_sharpe_mean']:.3f} ± {aggregate['wgt_sharpe_std']:.3f} (net: {aggregate['wgt_sharpe_net_mean']:.3f} ± {aggregate['wgt_sharpe_net_std']:.3f})")
    print(f"    Cumulative Return: {aggregate['wgt_return_cumulative']*100:.2f}% (net: {aggregate['wgt_return_net_cumulative']*100:.2f}%)")
    print(f"    Positive Sharpe_net in {aggregate['n_positive_wgt_sharpe_net']}/{n_folds} folds")
    print(f"\n  Total Trades: {total_trades_all}")
    
    return {
        "config": config,
        "tickers": tickers,
        "folds": [res for res in fold_results],
        "aggregate": aggregate,
    }


def save_walk_forward_report(wf_results: Dict[str, Any], output_path: str):
    """Save walk-forward validation results to JSON."""
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "engine_version": "V1.2",
            "mode": "walkforward",
            "tickers": wf_results["tickers"],
            "config": wf_results["config"],
            "thresholds": {
                "nn_buy_threshold": NN_BUY_THRESHOLD,
                "nn_sell_threshold": NN_SELL_THRESHOLD,
            },
            "cost_model": {
                "cost_bp_per_side": COST_BP_PER_SIDE,
            },
            "tda_mode": "extended" if USE_EXTENDED_TDA else "basic",
            "n_features": N_FEATURES,
        },
        "aggregate": wf_results["aggregate"],
        "folds": wf_results["folds"],
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nWalk-forward report saved to: {output_path}")


# =============================================================================
# V1.2: EXPANDED UNIVERSE
# =============================================================================

def run_expanded_universe(
    tickers: List[str] = None,
    total_cash: float = 100000.0,
) -> Dict[str, Any]:
    """
    Run backtest on expanded universe (V1.2).
    
    Uses the best-performing scenario configuration from robustness analysis:
      train_2022_2023_test_2024_2025
    
    This mode tests breadth: can we scale to 15-20 tickers?
    """
    if tickers is None:
        tickers = EXPANDED_TICKERS
    
    print("\n" + "=" * 80)
    print("V1.2 EXPANDED UNIVERSE MODE")
    print("=" * 80)
    print(f"\nTickers ({len(tickers)}): {tickers}")
    
    # Use best scenario config from V1.1 robustness analysis
    scenario = {
        "name": "expanded_universe_2022_2023_train_2024_2025_test",
        "train_start": "2022-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2025-12-31",
    }
    
    print(f"\nScenario: {scenario['name']}")
    print(f"  Train: {scenario['train_start']} to {scenario['train_end']}")
    print(f"  Test: {scenario['test_start']} to {scenario['test_end']}")
    
    result = run_scenario(scenario, tickers=tickers, total_cash=total_cash, verbose=True)
    
    # Print expanded universe summary
    print("\n" + "=" * 100)
    print("EXPANDED UNIVERSE SUMMARY (V1.2)")
    print("=" * 100)
    
    eq = result["portfolio_equal_weight"]
    wgt = result["portfolio_performance_weighted"]
    
    print(f"\n  Universe: {len(tickers)} tickers")
    print(f"\n  Equal-Weight Portfolio:")
    print(f"    Sharpe: {eq.get('sharpe_ratio', 0):.3f} (net: {eq.get('sharpe_ratio_net', 0):.3f})")
    print(f"    Return: {eq.get('total_return', 0)*100:.2f}% (net: {eq.get('total_return_net', 0)*100:.2f}%)")
    print(f"    Max DD: {eq.get('max_drawdown', 0)*100:.2f}%")
    print(f"    Trades: {eq.get('num_trades', 0)}, Turnover: {eq.get('turnover', 0):.2f}x")
    
    print(f"\n  Performance-Weighted Portfolio:")
    print(f"    Sharpe: {wgt.get('sharpe_ratio', 0):.3f} (net: {wgt.get('sharpe_ratio_net', 0):.3f})")
    print(f"    Return: {wgt.get('total_return', 0)*100:.2f}% (net: {wgt.get('total_return_net', 0)*100:.2f}%)")
    print(f"    Risk Scale: {result.get('risk_scale', 1.0):.2f}, Cash: {result.get('cash_weight', 0)*100:.0f}%")
    
    # Show top 5 weights
    weights = result.get("weights", {})
    sorted_weights = sorted(weights.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top 5 Weights: {{{', '.join([f'{t}: {w*100:.0f}%' for t, w in sorted_weights if w > 0.01])}}}")
    
    return result


def save_expanded_universe_report(result: Dict[str, Any], output_path: str):
    """Save expanded universe results to JSON."""
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "engine_version": "V1.2",
            "mode": "expanded_universe",
            "tickers": list(result.get("per_asset", {}).keys()),
            "n_tickers": len(result.get("per_asset", {})),
            "train_period": f"{result.get('train_start', 'N/A')} to {result.get('train_end', 'N/A')}",
            "test_period": f"{result.get('test_start', 'N/A')} to {result.get('test_end', 'N/A')}",
            "thresholds": {
                "nn_buy_threshold": NN_BUY_THRESHOLD,
                "nn_sell_threshold": NN_SELL_THRESHOLD,
            },
            "cost_model": {
                "cost_bp_per_side": COST_BP_PER_SIDE,
            },
            "tda_mode": "extended" if USE_EXTENDED_TDA else "basic",
            "n_features": N_FEATURES,
        },
        "per_asset": result.get("per_asset", {}),
        "portfolio_equal_weight": result.get("portfolio_equal_weight", {}),
        "portfolio_performance_weighted": result.get("portfolio_performance_weighted", {}),
        "weights": result.get("weights", {}),
        "risk_scale": result.get("risk_scale", 1.0),
        "cash_weight": result.get("cash_weight", 0.0),
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nExpanded universe report saved to: {output_path}")


# =============================================================================
# V1.2: REGIME LABELING (Analysis-Only)
# =============================================================================

def label_regimes_from_returns_and_tda(
    returns: np.ndarray,
    tda_entropy: np.ndarray = None,
    lookback: int = 20,
) -> np.ndarray:
    """
    Simple regime labeling based on returns and optional TDA entropy.
    
    V1.2: Analysis-only helper for interpretability.
    
    Regimes:
      - "trend": abs(rolling_mean) > 1 std and low volatility
      - "mean_reversion": low absolute mean, moderate volatility
      - "choppy": high volatility, no clear direction
      - "high_vol": volatility > 2 std (crisis/uncertainty)
    
    Args:
        returns: Array of daily returns
        tda_entropy: Optional TDA entropy signal for enhanced labeling
        lookback: Rolling window for statistics
    
    Returns:
        Array of string labels, same length as returns
    """
    n = len(returns)
    labels = np.array(["unknown"] * n, dtype=object)
    
    if n < lookback + 5:
        return labels
    
    # Compute rolling statistics
    rolling_mean = pd.Series(returns).rolling(lookback).mean().values
    rolling_std = pd.Series(returns).rolling(lookback).std().values
    
    # Global volatility baseline
    global_std = np.nanstd(rolling_std)
    global_mean_std = np.nanmean(rolling_std)
    
    for i in range(lookback, n):
        rm = rolling_mean[i]
        rs = rolling_std[i]
        
        # High volatility regime
        if rs > global_mean_std + 2 * global_std:
            labels[i] = "high_vol"
        # Trend regime: strong directional move with moderate vol
        elif abs(rm) > 0.001 and rs < global_mean_std + 0.5 * global_std:
            labels[i] = "trend"
        # Mean-reversion: low absolute mean
        elif abs(rm) < 0.0005 and rs < global_mean_std + global_std:
            labels[i] = "mean_reversion"
        # Choppy: everything else
        else:
            labels[i] = "choppy"
    
    # Fill initial values with first valid label
    first_valid_idx = lookback
    if first_valid_idx < n:
        labels[:first_valid_idx] = labels[first_valid_idx]
    
    return labels


def analyze_performance_by_regime(
    per_asset_results: Dict[str, Any],
    test_data_per_ticker: Dict,
) -> Dict[str, Any]:
    """
    Analyze strategy performance broken down by regime.
    
    V1.2: For interpretability, not live trading.
    
    Returns dict with per-regime metrics.
    """
    print("\n  Regime Analysis (V1.2 - Interpretability):")
    
    regime_stats = {
        "trend": {"count": 0, "trades": 0},
        "mean_reversion": {"count": 0, "trades": 0},
        "choppy": {"count": 0, "trades": 0},
        "high_vol": {"count": 0, "trades": 0},
        "unknown": {"count": 0, "trades": 0},
    }
    
    for ticker, data in test_data_per_ticker.items():
        df = data["df"]
        if len(df) < 25:
            continue
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        close = df[close_col].values
        returns = np.diff(np.log(close + 1e-10))
        returns = np.concatenate([[0], returns])
        
        labels = label_regimes_from_returns_and_tda(returns)
        
        for regime in regime_stats:
            regime_stats[regime]["count"] += np.sum(labels == regime)
    
    total_days = sum(r["count"] for r in regime_stats.values())
    
    print(f"    Regime Distribution (across all tickers):")
    for regime, stats in regime_stats.items():
        pct = stats["count"] / total_days * 100 if total_days > 0 else 0
        print(f"      {regime:15s}: {stats['count']:5d} days ({pct:5.1f}%)")
    
    return regime_stats


def run_all_scenarios() -> Dict[str, Any]:
    """
    Runs all SCENARIOS and returns a dict keyed by scenario name.
    V1.1: Includes cost-aware and risk-aware metrics in summary.
    """
    results_by_scenario = {}

    for i, scenario in enumerate(SCENARIOS):
        print(f"\n\n{'#'*80}")
        print(f"# SCENARIO {i+1}/{len(SCENARIOS)}: {scenario['name']}")
        print(f"{'#'*80}")
        
        scenario_result = run_scenario(scenario)
        name = scenario_result.get("name", scenario["name"])
        results_by_scenario[name] = scenario_result

    # Cross-scenario summary (V1.1: includes net metrics)
    print("\n\n" + "=" * 110)
    print("CROSS-SCENARIO PORTFOLIO SUMMARY (V1.1 - Cost & Risk Aware)")
    print("=" * 110)
    print(f"{'Scenario':<35} {'EQ_Sh':>7} {'EQ_Sh_N':>8} {'WGT_Sh':>7} {'WGT_Sh_N':>9} {'EQ_Ret%':>8} {'WGT_Ret%':>9} {'Risk':>5}")
    print("-" * 110)
    
    for name, res in results_by_scenario.items():
        eq = res["portfolio_equal_weight"]
        wgt = res["portfolio_performance_weighted"]
        risk = res.get("risk_scale", 1.0)
        print(f"{name:<35} {eq['sharpe_ratio']:>7.2f} {eq.get('sharpe_ratio_net',0):>8.2f} "
              f"{wgt['sharpe_ratio']:>7.2f} {wgt.get('sharpe_ratio_net',0):>9.2f} "
              f"{eq['total_return']*100:>7.2f}% {wgt['total_return']*100:>8.2f}% {risk:>5.2f}")
    
    print("-" * 110)
    
    # Weights stability summary
    print("\n" + "=" * 90)
    print("WEIGHTS STABILITY ACROSS SCENARIOS")
    print("=" * 90)
    print(f"{'Scenario':<40} {'SPY':>8} {'QQQ':>8} {'IWM':>8} {'XLF':>8} {'XLK':>8}")
    print("-" * 90)
    
    for name, res in results_by_scenario.items():
        w = res.get("weights", {})
        print(f"{name:<40} {w.get('SPY',0)*100:>7.0f}% {w.get('QQQ',0)*100:>7.0f}% "
              f"{w.get('IWM',0)*100:>7.0f}% {w.get('XLF',0)*100:>7.0f}% {w.get('XLK',0)*100:>7.0f}%")
    
    print("-" * 90)
    
    return results_by_scenario


def save_robustness_report(results_by_scenario: Dict[str, Any], output_path: str):
    """Save robustness analysis results to JSON (V1.1: includes cost and risk metrics)."""
    
    # Build cross-scenario summary
    summary = []
    for name, res in results_by_scenario.items():
        eq = res["portfolio_equal_weight"]
        wgt = res["portfolio_performance_weighted"]
        summary.append({
            "scenario": name,
            "train_period": f"{res['train_start']} to {res['train_end']}",
            "test_period": f"{res['test_start']} to {res['test_end']}",
            # Gross metrics
            "eq_sharpe": eq.get("sharpe_ratio", 0),
            "wgt_sharpe": wgt.get("sharpe_ratio", 0),
            "eq_return": eq.get("total_return", 0),
            "wgt_return": wgt.get("total_return", 0),
            # V1.1: Net (cost-adjusted) metrics
            "eq_sharpe_net": eq.get("sharpe_ratio_net", 0),
            "wgt_sharpe_net": wgt.get("sharpe_ratio_net", 0),
            "eq_return_net": eq.get("total_return_net", 0),
            "wgt_return_net": wgt.get("total_return_net", 0),
            # V1.1: Risk overlay
            "risk_scale": res.get("risk_scale", 1.0),
            "cash_weight": res.get("cash_weight", 0.0),
            # Other
            "num_trades": eq.get("num_trades", 0),
            "turnover": eq.get("turnover", 0),
            "weights": res.get("weights", {}),
        })
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "engine_version": "V1.1",
            "tickers": TICKERS,
            "scenarios": list(results_by_scenario.keys()),
            "thresholds": {
                "nn_buy_threshold": NN_BUY_THRESHOLD,
                "nn_sell_threshold": NN_SELL_THRESHOLD,
            },
            "cost_model": {
                "cost_bp_per_side": COST_BP_PER_SIDE,
                "note": "Extra slippage/impact on top of 0.1% broker commission"
            },
        },
        "summary": summary,
        "scenarios": results_by_scenario,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nRobustness report saved to: {output_path}")


# =============================================================================
# ABLATION EXPERIMENT (V1.3)
# =============================================================================

def run_ablation_experiment(scenario: Dict[str, str] = None,
                            tickers: List[str] = None) -> Dict[str, Any]:
    """
    Run ablation experiments comparing TDA feature modes.
    
    V1.3: Compares v1.1 (4 TDA), v1.2 (10 TDA), v1.3 (20 TDA) features.
    
    Args:
        scenario: Optional scenario dict (defaults to baseline)
        tickers: Optional ticker list (defaults to TICKERS)
        
    Returns:
        Dict with ablation results per feature mode
    """
    global TDA_FEATURE_MODE, N_FEATURES, USE_EXTENDED_TDA
    
    if scenario is None:
        scenario = SCENARIOS[1]  # train_2022_2023_test_2024_2025
    
    if tickers is None:
        tickers = TICKERS
    
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT: TDA Feature Mode Comparison (V1.3)")
    print("=" * 80)
    print(f"  Scenario: {scenario['name']}")
    print(f"  Train: {scenario['train_start']} to {scenario['train_end']}")
    print(f"  Test: {scenario['test_start']} to {scenario['test_end']}")
    print(f"  Tickers: {tickers}")
    
    feature_modes = ['v1.1', 'v1.2', 'v1.3']
    ablation_results = {}
    
    for mode in feature_modes:
        print(f"\n{'='*60}")
        print(f"Running ablation with TDA_FEATURE_MODE = '{mode}'")
        print(f"{'='*60}")
        
        # Update global config
        n_tda = TDA_FEATURE_COUNTS[mode]
        n_features = n_tda + 2
        
        print(f"  TDA features: {n_tda}")
        print(f"  Total N_FEATURES: {n_features}")
        
        # Initialize generators with this mode
        tda_gen = TDAFeatureGenerator(window=20, embedding_dim=3, feature_mode=mode)
        preprocessor = DataPreprocessor(
            sequence_length=SEQUENCE_LENGTH, 
            use_extended_tda=(mode in ('v1.2', 'v1.3'))
        )
        
        # Build dataset
        train_X, train_y, test_data_per_ticker = build_multiasset_dataset(
            tickers=tickers,
            train_start=scenario['train_start'],
            train_end=scenario['train_end'],
            test_start=scenario['test_start'],
            test_end=scenario['test_end'],
            tda_gen=tda_gen,
            preprocessor=preprocessor
        )
        
        if len(train_X) == 0 or not test_data_per_ticker:
            print(f"  ⚠️ No data for mode {mode}")
            ablation_results[mode] = {
                'tda_features': n_tda,
                'n_features': n_features,
                'sharpe_net': 0.0,
                'return_net': 0.0,
                'trades': 0,
                'error': 'No data'
            }
            continue
        
        # Create and train model with correct n_features
        model = NeuralNetPredictor(
            sequence_length=SEQUENCE_LENGTH, 
            n_features=n_features, 
            lstm_units=LSTM_UNITS
        )
        model.compile_model(learning_rate=LEARNING_RATE, use_entropy_penalty=True)
        _ = model(train_X[:1])
        
        print(f"  Training model on {len(train_X)} samples...")
        history = train_model(
            model, train_X, train_y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            track_output_spread=False
        )
        
        # Run backtest
        per_asset_results, all_daily_returns = run_multiasset_backtest(
            test_data_per_ticker=test_data_per_ticker,
            model=model,
            tda_gen=tda_gen,
            preprocessor=preprocessor,
            total_cash=100000
        )
        
        # Compute portfolio metrics using the unified function
        eq_weight_metrics = compute_portfolio_metrics(
            per_asset_results, all_daily_returns, 100000, 
            weights=None, label="Equal-Weight"
        )
        perf_weights = compute_performance_weights(per_asset_results)
        wgt_metrics = compute_portfolio_metrics(
            per_asset_results, all_daily_returns, 100000,
            weights=perf_weights, label="Performance-Weighted"
        )
        
        ablation_results[mode] = {
            'tda_features': n_tda,
            'n_features': n_features,
            'sharpe_net': wgt_metrics.get('sharpe_ratio_net', 0),
            'return_net': wgt_metrics.get('total_return_net', 0),
            'sharpe_gross': wgt_metrics.get('sharpe_ratio', 0),
            'return_gross': wgt_metrics.get('total_return', 0),
            'trades': wgt_metrics.get('num_trades', 0),
            'turnover': wgt_metrics.get('turnover', 0),
            'per_asset': per_asset_results,
        }
        
        print(f"\n  Results for {mode}:")
        print(f"    Sharpe_net: {wgt_metrics.get('sharpe_ratio_net', 0):.3f}")
        print(f"    Return_net: {wgt_metrics.get('total_return_net', 0)*100:.2f}%")
        print(f"    Trades: {wgt_metrics.get('num_trades', 0)}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("ABLATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Mode':<10} {'TDA':>6} {'N_Feat':>8} {'SharpeNet':>12} {'RetNet%':>10} {'Trades':>8}")
    print("-" * 60)
    for mode in feature_modes:
        r = ablation_results.get(mode, {})
        print(f"{mode:<10} {r.get('tda_features', 0):>6} {r.get('n_features', 0):>8} "
              f"{r.get('sharpe_net', 0):>12.3f} {r.get('return_net', 0)*100:>10.2f} "
              f"{r.get('trades', 0):>8}")
    
    # Determine best mode
    best_mode = max(ablation_results.items(), 
                   key=lambda x: x[1].get('sharpe_net', -999))[0]
    print(f"\n  Best TDA Mode: {best_mode} (by Sharpe_net)")
    
    return ablation_results


def run_ablation_with_regimes(scenario: Dict[str, str] = None,
                               tickers: List[str] = None) -> Dict[str, Any]:
    """
    Run ablation experiments with regime analysis.
    
    V1.3: Adds per-regime performance breakdown.
    """
    if scenario is None:
        scenario = SCENARIOS[1]
    
    if tickers is None:
        tickers = TICKERS
    
    # First run ablation
    ablation_results = run_ablation_experiment(scenario, tickers)
    
    # Now compute regime analysis for best mode (v1.3)
    print("\n" + "=" * 80)
    print("REGIME ANALYSIS (V1.3)")
    print("=" * 80)
    
    # Initialize generators with v1.3
    tda_gen = TDAFeatureGenerator(window=20, embedding_dim=3, feature_mode='v1.3')
    regime_labeler = RegimeLabeler(window=20)
    
    regime_results = {}
    
    for ticker in tickers:
        print(f"\n  Analyzing regimes for {ticker}...")
        
        # Download data
        df = download_ticker_data(ticker, scenario['test_start'], scenario['test_end'])
        if df.empty:
            continue
        
        # Generate TDA features
        tda_features = tda_gen.generate_features(df)
        
        # Align TDA with price data
        start_idx = len(df) - len(tda_features)
        df_aligned = df.iloc[start_idx:].copy()
        tda_features.index = df_aligned.index
        
        # Label regimes
        regimes = regime_labeler.label_regimes(df_aligned, tda_features)
        
        # Compute regime summary
        summary = regime_labeler.get_regime_summary(regimes)
        regime_results[ticker] = summary
        
        print(f"    Regimes: {summary['regimes']}")
    
    ablation_results['regimes'] = regime_results
    
    return ablation_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main orchestration for multi-asset TDA+NN strategy V1.3.
    
    Modes:
      - "baseline": Run single baseline scenario (train 2022-2023, test 2024-2025)
      - "robustness": Run all scenarios for full robustness analysis
      - "walkforward": V1.2 walk-forward validation
      - "expanded_universe": V1.2 test with 18 tickers
      - "ablation": V1.3 TDA feature ablation experiment
    """
    print("=" * 80)
    print("MULTI-ASSET TDA+NN PORTFOLIO STRATEGY – ENGINE V1.3")
    print("=" * 80)
    print(f"\nEngine V1.3 Features:")
    print(f"  • V1.3 NEW: Enriched TDA features (20 total: top_k, count_large, wasserstein)")
    print(f"  • V1.3 NEW: Regime labeling (trend_up, trend_down, high_vol, choppy)")
    print(f"  • V1.3 NEW: Feature ablation experiments (v1.1/v1.2/v1.3)")
    print(f"  • V1.2-data: Polygon/Massive as primary data provider")
    print(f"  • V1.2: Walk-forward validation, expanded universe")
    print(f"  • V1.1: Cost-aware metrics, risk overlay")
    
    # V1.2-data: Validate data provider configuration
    print(f"\nData Provider Configuration:")
    print(f"  Provider: {DATA_PROVIDER}")
    print(f"  Timeframe: {DEFAULT_TIMEFRAME}")
    
    if DATA_PROVIDER.lower() == "polygon":
        try:
            validate_provider("polygon", POLYGON_API_KEY_ENV)
            print(f"  ✓ Polygon API key loaded from ${POLYGON_API_KEY_ENV}")
        except Exception as e:
            print(f"  ✗ Polygon provider initialization failed: {e}")
            print(f"\n  To use Polygon, set the API key:")
            print(f"    export {POLYGON_API_KEY_ENV}=your_otrep_key")
            print(f"\n  Or switch to yfinance by setting DATA_PROVIDER='yfinance'")
            return {}
    else:
        print(f"  ✓ Using {DATA_PROVIDER} (no API key required)")
    
    print(f"\nConfiguration:")
    print(f"  Mode: {MODE}")
    print(f"  TDA Feature Mode: {TDA_FEATURE_MODE} ({TDA_FEATURE_COUNTS[TDA_FEATURE_MODE]} TDA features)")
    print(f"  N_FEATURES: {N_FEATURES}")
    print(f"  Tickers: {TICKERS if MODE != 'expanded_universe' else EXPANDED_TICKERS}")
    print(f"  Thresholds: Buy > {NN_BUY_THRESHOLD}, Sell < {NN_SELL_THRESHOLD}")
    print(f"  Confidence Sizing: {USE_CONFIDENCE_SIZING}")
    print(f"  Cost Model: {COST_BP_PER_SIDE} bp/side")
    
    if MODE == "baseline":
        # Run single baseline scenario
        print(f"\nRunning BASELINE scenario only...")
        baseline_scenario = SCENARIOS[1]  # train_2022_2023_test_2024_2025 (best from V1.1)
        result = run_scenario(baseline_scenario)
        
        # Save to single-scenario JSON
        save_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "engine_version": "V1.3",
                "mode": "baseline",
                "data_provider": DATA_PROVIDER,
                "timeframe": DEFAULT_TIMEFRAME,
                "tickers": TICKERS,
                "tda_feature_mode": TDA_FEATURE_MODE,
                "tda_features": {
                    "version": TDA_FEATURE_MODE,
                    "n_tda_features": TDA_FEATURE_COUNTS[TDA_FEATURE_MODE],
                    "n_total_features": N_FEATURES,
                    "description": "H0/H1 persistence (max, mean, sum, entropy, top_k, count_large, wasserstein)"
                        if TDA_FEATURE_MODE == 'v1.3' else "H0/H1 persistence (basic)",
                },
                "train_period": f"{baseline_scenario['train_start']} to {baseline_scenario['train_end']}",
                "test_period": f"{baseline_scenario['test_start']} to {baseline_scenario['test_end']}",
                "thresholds": {
                    "nn_buy_threshold": NN_BUY_THRESHOLD,
                    "nn_sell_threshold": NN_SELL_THRESHOLD,
                },
                "cost_model": {
                    "cost_bp_per_side": COST_BP_PER_SIDE,
                }
            },
            "per_asset": result["per_asset"],
            "portfolio_equal_weight": result["portfolio_equal_weight"],
            "portfolio_performance_weighted": result["portfolio_performance_weighted"],
            "weights": result["weights"],
            "risk_scale": result.get("risk_scale", 1.0),
            "cash_weight": result.get("cash_weight", 0.0),
            "threshold_sensitivity": result.get("threshold_sensitivity", {}),
        }
        
        with open(RESULTS_JSON_PATH, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nBaseline results saved to: {RESULTS_JSON_PATH}")
        
        # V1.3: Print summary with data provider info
        print(f"\n" + "=" * 60)
        print(f"BASELINE COMPLETE (V1.3)")
        print(f"=" * 60)
        print(f"  Data Provider: {DATA_PROVIDER}")
        print(f"  TDA Mode: {TDA_FEATURE_MODE} ({TDA_FEATURE_COUNTS[TDA_FEATURE_MODE]} TDA features)")
        print(f"  Timeframe: {DEFAULT_TIMEFRAME}")
        wgt = result.get("portfolio_performance_weighted", {})
        print(f"  WGT Sharpe_net: {wgt.get('sharpe_ratio_net', 0):.3f}")
        print(f"  WGT Return_net: {wgt.get('total_return_net', 0)*100:.2f}%")
        print(f"  Total Trades: {wgt.get('num_trades', 0)}")
        print(f"  Turnover: {wgt.get('turnover', 0):.2f}x")
        
        return {"baseline": result}
    
    elif MODE == "robustness":
        print(f"\nRunning ROBUSTNESS analysis ({len(SCENARIOS)} scenarios)...")
        for s in SCENARIOS:
            print(f"    - {s['name']}")
        
        # Run all scenarios
        results_by_scenario = run_all_scenarios()
        
        # Save robustness report
        save_robustness_report(results_by_scenario, ROBUSTNESS_JSON_PATH)
        
        # Final interpretation
        print("\n" + "=" * 80)
        print("ROBUSTNESS ANALYSIS COMPLETE (V1.2)")
        print("=" * 80)
        
        # Count positive scenarios for net metrics
        positive_wgt_sharpe_net = sum(1 for r in results_by_scenario.values() 
                                       if r["portfolio_performance_weighted"].get("sharpe_ratio_net", 0) > 0)
        positive_wgt_return_net = sum(1 for r in results_by_scenario.values() 
                                       if r["portfolio_performance_weighted"].get("total_return_net", 0) > 0)
        
        print(f"\n  Performance-Weighted Portfolio (NET of costs):")
        print(f"    Positive Sharpe_net in {positive_wgt_sharpe_net}/{len(SCENARIOS)} scenarios")
        print(f"    Positive Return_net in {positive_wgt_return_net}/{len(SCENARIOS)} scenarios")
        
        # Find best and worst scenarios by net Sharpe
        best_scenario = max(results_by_scenario.items(), 
                            key=lambda x: x[1]["portfolio_performance_weighted"].get("sharpe_ratio_net", -999))
        worst_scenario = min(results_by_scenario.items(), 
                             key=lambda x: x[1]["portfolio_performance_weighted"].get("sharpe_ratio_net", 999))
        
        print(f"\n  Best WGT Sharpe_net: {best_scenario[0]} ({best_scenario[1]['portfolio_performance_weighted'].get('sharpe_ratio_net', 0):.2f})")
        print(f"  Worst WGT Sharpe_net: {worst_scenario[0]} ({worst_scenario[1]['portfolio_performance_weighted'].get('sharpe_ratio_net', 0):.2f})")
        
        return results_by_scenario
    
    elif MODE == "walkforward":
        print(f"\nRunning WALK-FORWARD validation (V1.2)...")
        print(f"  Config: {WALK_FORWARD_CONFIG}")
        
        # Run walk-forward
        wf_results = run_walk_forward(
            tickers=TICKERS,
            config=WALK_FORWARD_CONFIG,
            total_cash=100000.0,
        )
        
        # Save report
        save_walk_forward_report(wf_results, WALKFORWARD_JSON_PATH)
        
        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION COMPLETE (V1.2)")
        print("=" * 80)
        
        agg = wf_results["aggregate"]
        print(f"\n  Summary:")
        print(f"    Folds: {agg['n_folds']}")
        print(f"    WGT Sharpe_net: {agg['wgt_sharpe_net_mean']:.3f} ± {agg['wgt_sharpe_net_std']:.3f}")
        print(f"    Positive Sharpe_net: {agg['n_positive_wgt_sharpe_net']}/{agg['n_folds']} folds")
        print(f"    Cumulative Return_net: {agg['wgt_return_net_cumulative']*100:.2f}%")
        
        return {"walkforward": wf_results}
    
    elif MODE == "expanded_universe":
        print(f"\nRunning EXPANDED UNIVERSE mode (V1.2)...")
        print(f"  Tickers ({len(EXPANDED_TICKERS)}): {EXPANDED_TICKERS}")
        
        # Run expanded universe
        result = run_expanded_universe(
            tickers=EXPANDED_TICKERS,
            total_cash=100000.0,
        )
        
        # Save report
        save_expanded_universe_report(result, EXPANDED_JSON_PATH)
        
        print("\n" + "=" * 80)
        print("EXPANDED UNIVERSE COMPLETE (V1.2)")
        print("=" * 80)
        
        wgt = result["portfolio_performance_weighted"]
        print(f"\n  Summary:")
        print(f"    Universe: {len(EXPANDED_TICKERS)} tickers")
        print(f"    WGT Sharpe_net: {wgt.get('sharpe_ratio_net', 0):.3f}")
        print(f"    WGT Return_net: {wgt.get('total_return_net', 0)*100:.2f}%")
        
        return {"expanded_universe": result}
    
    elif MODE == "ablation":
        print(f"\nRunning ABLATION experiment (V1.3)...")
        print(f"  Comparing TDA feature modes: v1.1, v1.2, v1.3")
        
        # Run ablation with regime analysis
        ablation_results = run_ablation_with_regimes()
        
        # Save ablation report
        ablation_path = f'{RESULTS_DIR}/ablation_report.json'
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "engine_version": "V1.3",
                "experiment": "TDA feature ablation",
                "tickers": TICKERS,
            },
            "ablation": {k: v for k, v in ablation_results.items() if k != 'regimes'},
            "regimes": ablation_results.get('regimes', {}),
        }
        with open(ablation_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nAblation report saved to: {ablation_path}")
        
        return {"ablation": ablation_results}
    
    else:
        print(f"\n  ⚠️  Unknown MODE: {MODE}")
        print(f"  Valid modes: baseline, robustness, walkforward, expanded_universe, ablation")
        return {}


if __name__ == "__main__":
    results = main()
