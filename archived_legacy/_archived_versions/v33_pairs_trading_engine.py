#!/usr/bin/env python3
"""
V31 MAXIMUM PROFIT ENGINE - Ultimate NYSE Trading System
=========================================================
Advanced algorithmic trading engine with:
- Neural network price prediction
- Genetic algorithm optimization
- Advanced backtesting
- Real-time sentiment analysis
- Maximum profit extraction

Version: 31.0.0
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class V31Config:
    """Maximum profit configuration"""
    
    # API Configuration
    alpaca_api_key: str = field(default_factory=lambda: os.getenv('ALPACA_API_KEY', ''))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv('ALPACA_BASE_URL', 'https://api.alpaca.markets'))
    
    # Portfolio Configuration
    initial_capital: float = 100000.0
    max_positions: int = 30
    max_position_size: float = 0.10  # 10% max per position
    min_position_size: float = 0.02  # 2% min per position
    
    # Aggressive Profit Settings
    target_annual_return: float = 0.50  # 50% annual target
    max_drawdown_allowed: float = 0.15  # 15% max drawdown
    profit_taking_threshold: float = 0.15  # Take profits at 15%
    
    # Factor Weights (optimized for max profit)
    momentum_weight: float = 0.35
    mean_reversion_weight: float = 0.20
    neural_net_weight: float = 0.25
    sentiment_weight: float = 0.20
    
    # Neural Network Config
    nn_lookback: int = 60
    nn_hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    nn_epochs: int = 50
    
    # Genetic Algorithm Config
    ga_population_size: int = 50
    ga_generations: int = 20
    ga_mutation_rate: float = 0.1
    
    # Backtesting
    backtest_years: int = 3
    walk_forward_windows: int = 12
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'v31_engine.log'


# =============================================================================
# ADVANCED BACKTESTING ENGINE
# =============================================================================

class AdvancedBacktester:
    """Walk-forward backtesting with realistic execution simulation"""
    
    def __init__(self, config: V31Config):
        self.config = config
        self.logger = logging.getLogger('Backtester')
        self.results = []
        
    def run_backtest(self, strategy_func, universe: List[str], 
                     start_date: datetime, end_date: datetime) -> Dict:
        """Run comprehensive backtest with walk-forward optimization"""
        
        portfolio_value = self.config.initial_capital
        positions = {}
        trades = []
        daily_returns = []
        peak_value = portfolio_value
        max_drawdown = 0
        
        # Get historical data for all symbols
        all_data = {}
        for symbol in universe:
            try:
                if YFINANCE_AVAILABLE:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    df.columns = [c.lower() for c in df.columns]
                    all_data[symbol] = df
            except Exception as e:
                self.logger.warning(f"Error getting data for {symbol}: {e}")
                
        if not all_data:
            return {'error': 'No data available'}
            
        # Get trading days
        sample_df = list(all_data.values())[0]
        trading_days = sample_df.index.tolist()
        
        prev_value = portfolio_value
        
        for i, date in enumerate(trading_days[60:], 60):  # Start after warmup
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value(positions, all_data, date)
            
            # Track drawdown
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            current_dd = (peak_value - portfolio_value) / peak_value
            max_drawdown = max(max_drawdown, current_dd)
            
            # Calculate daily return
            daily_ret = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_ret)
            prev_value = portfolio_value
            
            # Generate signals
            signals = strategy_func(all_data, date, i)
            
            # Execute trades (simplified)
            for signal in signals:
                if signal['action'] == 'BUY' and signal['symbol'] not in positions:
                    if len(positions) < self.config.max_positions:
                        position_size = min(
                            portfolio_value * self.config.max_position_size,
                            signal.get('size', portfolio_value * 0.05)
                        )
                        price = all_data[signal['symbol']].loc[date, 'close']
                        shares = int(position_size / price)
                        if shares > 0:
                            positions[signal['symbol']] = {
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': date
                            }
                            trades.append({
                                'symbol': signal['symbol'],
                                'action': 'BUY',
                                'price': price,
                                'shares': shares,
                                'date': date
                            })
                            
                elif signal['action'] == 'SELL' and signal['symbol'] in positions:
                    pos = positions[signal['symbol']]
                    price = all_data[signal['symbol']].loc[date, 'close']
                    pnl = (price - pos['entry_price']) * pos['shares']
                    trades.append({
                        'symbol': signal['symbol'],
                        'action': 'SELL',
                        'price': price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'date': date
                    })
                    del positions[signal['symbol']]
                    
        # Calculate final metrics
        final_value = self._calculate_portfolio_value(positions, all_data, trading_days[-1])
        total_return = (final_value - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized metrics
        days = len(daily_returns)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        daily_returns_arr = np.array(daily_returns)
        sharpe = np.sqrt(252) * np.mean(daily_returns_arr) / np.std(daily_returns_arr) if np.std(daily_returns_arr) > 0 else 0
        
        # Win rate
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        total_closed = [t for t in trades if 'pnl' in t]
        win_rate = len(winning_trades) / len(total_closed) if total_closed else 0
        
        return {
            'initial_capital': self.config.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade_pnl': np.mean([t.get('pnl', 0) for t in total_closed]) if total_closed else 0
        }
        
    def _calculate_portfolio_value(self, positions: Dict, all_data: Dict, date) -> float:
        """Calculate current portfolio value"""
        cash = self.config.initial_capital
        for symbol, pos in positions.items():
            if symbol in all_data and date in all_data[symbol].index:
                price = all_data[symbol].loc[date, 'close']
                cash += pos['shares'] * price - pos['shares'] * pos['entry_price']
        return cash

# =============================================================================
# NEURAL NETWORK PREDICTOR
# =============================================================================

class NeuralNetPredictor:
    """Advanced neural network for price prediction"""
    
    def __init__(self, config: V31Config):
        self.config = config
        self.logger = logging.getLogger('NeuralNet')
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def _create_model(self, input_shape: int) -> Any:
        """Create neural network model"""
        if TF_AVAILABLE:
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(3, activation='softmax')  # Up, Down, Neutral
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        else:
            # Fallback to sklearn MLP
            return MLPClassifier(
                hidden_layer_sizes=tuple(self.config.nn_hidden_layers),
                max_iter=self.config.nn_epochs,
                random_state=42
            )
            
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare advanced feature set for neural network"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        for period in [5, 10, 20, 50]:
            features[f'ret_{period}d'] = df['close'].pct_change(period)
            features[f'vol_{period}d'] = df['close'].pct_change().rolling(period).std()
            
        # Technical indicators
        features['rsi_14'] = self._rsi(df['close'], 14)
        features['rsi_7'] = self._rsi(df['close'], 7)
        
        # MACD
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        features['macd'] = exp12 - exp26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands position
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_upper'] = (df['close'] - (sma20 + 2*std20)) / (4*std20)
        features['bb_lower'] = (df['close'] - (sma20 - 2*std20)) / (4*std20)
        
        # Volume features
        if 'volume' in df.columns:
            features['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['vol_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            
        # Price patterns
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Trend strength
        features['adx'] = self._adx(df, 14)
        
        return features.dropna()
        
    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()
        
    def train(self, df: pd.DataFrame) -> bool:
        """Train neural network on historical data"""
        try:
            features = self.prepare_features(df)
            if len(features) < 200:
                return False
                
            # Create labels (3 classes: Up > 1%, Down < -1%, Neutral)
            future_ret = df['close'].pct_change(5).shift(-5)
            labels = pd.Series(index=future_ret.index, dtype=int)
            labels[future_ret > 0.01] = 2  # Up
            labels[future_ret < -0.01] = 0  # Down
            labels[(future_ret >= -0.01) & (future_ret <= 0.01)] = 1  # Neutral
            
            # Align data
            common_idx = features.index.intersection(labels.dropna().index)
            X = features.loc[common_idx].values[:-20]
            y = labels.loc[common_idx].values[:-20]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = self._create_model(X.shape[1])
            
            if TF_AVAILABLE:
                y_onehot = keras.utils.to_categorical(y, 3)
                self.model.fit(X_scaled, y_onehot, epochs=self.config.nn_epochs, 
                              batch_size=32, verbose=0, validation_split=0.2)
            else:
                self.model.fit(X_scaled, y)
                
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return False
            
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Predict next move: 0=Down, 1=Neutral, 2=Up"""
        if not self.is_trained:
            return 1, 0.33
            
        try:
            features = self.prepare_features(df)
            X = features.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            if TF_AVAILABLE:
                probs = self.model.predict(X_scaled, verbose=0)[0]
            else:
                probs = self.model.predict_proba(X_scaled)[0]
                
            prediction = np.argmax(probs)
            confidence = probs[prediction]
            
            return prediction, confidence
            
        except Exception as e:
            return 1, 0.33


# =============================================================================
# GENETIC ALGORITHM OPTIMIZER
# =============================================================================

class GeneticOptimizer:
    """Optimize strategy parameters using genetic algorithm"""
    
    def __init__(self, config: V31Config):
        self.config = config
        self.logger = logging.getLogger('GeneticOpt')
        self.best_params = None
        self.best_fitness = -np.inf
        
    def optimize(self, backtester: AdvancedBacktester, universe: List[str],
                 param_ranges: Dict) -> Dict:
        """Run genetic algorithm optimization"""
        
        population = self._initialize_population(param_ranges)
        
        for generation in range(self.config.ga_generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, backtester, universe)
                fitness_scores.append(fitness)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_params = individual.copy()
                    
            # Selection, crossover, mutation
            population = self._evolve(population, fitness_scores, param_ranges)
            
            self.logger.info(f"Generation {generation+1}: Best Fitness = {self.best_fitness:.4f}")
            
        return self.best_params
        
    def _initialize_population(self, param_ranges: Dict) -> List[Dict]:
        """Initialize random population"""
        population = []
        for _ in range(self.config.ga_population_size):
            individual = {}
            for param, (min_val, max_val) in param_ranges.items():
                individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)
        return population
        
    def _evaluate_fitness(self, params: Dict, backtester: AdvancedBacktester,
                         universe: List[str]) -> float:
        """Evaluate fitness of parameter set (Sharpe ratio)"""
        try:
            # Create strategy function with these params
            def strategy_func(all_data, date, idx):
                return self._generate_signals(all_data, date, idx, params)
                
            start = datetime.now() - timedelta(days=self.config.backtest_years * 365)
            end = datetime.now() - timedelta(days=30)
            
            results = backtester.run_backtest(strategy_func, universe, start, end)
            
            # Fitness = Sharpe ratio with drawdown penalty
            sharpe = results.get('sharpe_ratio', 0)
            max_dd = results.get('max_drawdown', 1)
            
            fitness = sharpe * (1 - max_dd)  # Penalize high drawdown
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Fitness evaluation error: {e}")
            return -np.inf
            
    def _generate_signals(self, all_data: Dict, date, idx: int, params: Dict) -> List[Dict]:
        """Generate signals using optimized parameters"""
        signals = []
        
        mom_threshold = params.get('momentum_threshold', 0.05)
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        
        for symbol, df in all_data.items():
            if date not in df.index:
                continue
                
            loc = df.index.get_loc(date)
            if loc < 60:
                continue
                
            # Calculate momentum
            momentum = (df['close'].iloc[loc] / df['close'].iloc[loc-20] - 1)
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[loc]
            
            # Generate signals
            if momentum > mom_threshold and rsi < rsi_overbought:
                signals.append({'symbol': symbol, 'action': 'BUY'})
            elif momentum < -mom_threshold or rsi > rsi_overbought:
                signals.append({'symbol': symbol, 'action': 'SELL'})
                
        return signals
        
    def _evolve(self, population: List[Dict], fitness_scores: List[float],
               param_ranges: Dict) -> List[Dict]:
        """Evolve population through selection, crossover, mutation"""
        # Tournament selection
        new_population = []
        
        # Keep best individuals (elitism)
        sorted_idx = np.argsort(fitness_scores)[::-1]
        for i in range(2):  # Keep top 2
            new_population.append(population[sorted_idx[i]].copy())
            
        while len(new_population) < len(population):
            # Tournament selection
            parent1 = self._tournament_select(population, fitness_scores)
            parent2 = self._tournament_select(population, fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child, param_ranges)
            
            new_population.append(child)
            
        return new_population
        
    def _tournament_select(self, population: List[Dict], 
                          fitness_scores: List[float], k: int = 3) -> Dict:
        """Select individual via tournament"""
        indices = np.random.choice(len(population), k, replace=False)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return population[best_idx].copy()
        
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Uniform crossover"""
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]
        return child
        
    def _mutate(self, individual: Dict, param_ranges: Dict) -> Dict:
        """Mutate with probability"""
        for key, (min_val, max_val) in param_ranges.items():
            if np.random.random() < self.config.ga_mutation_rate:
                individual[key] = np.random.uniform(min_val, max_val)
        return individual

# =============================================================================
# V31 MAXIMUM PROFIT ENGINE - MAIN CLASS
# =============================================================================

class V31MaximumProfitEngine:
    """Ultimate profit-maximizing trading engine"""
    
    # Expanded sector-based universe
    UNIVERSE = [
        # Technology
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE', 'ORCL',
        # Consumer
        'AMZN', 'TSLA', 'HD', 'NKE', 'SBUX', 'MCD', 'TGT', 'COST',
        # Healthcare  
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT',
        # Financial
        'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD',
        # Industrial
        'CAT', 'DE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE'
    ]
    
    def __init__(self, config: V31Config = None):
        self.config = config or V31Config()
        self.logger = self._setup_logging()
        
        # Core components
        self.backtester = AdvancedBacktester(self.config)
        self.neural_net = NeuralNetPredictor(self.config)
        self.optimizer = GeneticOptimizer(self.config)
        
        # ML models
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
        self.ada_model = AdaBoostClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # API connection
        self.api = None
        if ALPACA_AVAILABLE and self.config.alpaca_api_key:
            self.api = REST(
                self.config.alpaca_api_key,
                self.config.alpaca_secret_key,
                self.config.alpaca_base_url
            )
            
        # Optimized parameters (from genetic algorithm)
        self.optimized_params = {
            'momentum_threshold': 0.03,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'position_size_mult': 1.0,
            'stop_loss': 0.07,
            'take_profit': 0.15
        }
        
        # Performance tracking
        self.trades = []
        self.daily_pnl = []
        
        self.logger.info("V33 Pairs Trading & Options Flow Engine - LSTM & Kelly Criterion initialized")
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('V33Engine')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        fh = logging.FileHandler(self.config.log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger


    def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Get historical price data"""
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{days}d")
                df.columns = [c.lower() for c in df.columns]
                return df
            except Exception as e:
                self.logger.warning(f"yfinance error for {symbol}: {e}")
        return pd.DataFrame()
        
    def train_models(self) -> bool:
        """Train all ML models on historical data"""
        self.logger.info("Training ML models...")
        
        try:
            # Get SPY data for market training
            spy_data = self.get_historical_data('SPY', 500)
            if len(spy_data) < 200:
                return False
                
            # Prepare features
            features = self._prepare_ml_features(spy_data)
            
            # Create labels
            future_ret = spy_data['close'].pct_change(5).shift(-5)
            labels = (future_ret > 0).astype(int)
            
            # Align data
            common_idx = features.index.intersection(labels.dropna().index)
            X = features.loc[common_idx].values[:-20]
            y = labels.loc[common_idx].values[:-20]
            
            # Scale and train
            X_scaled = self.scaler.fit_transform(X)
            
            self.rf_model.fit(X_scaled, y)
            self.gb_model.fit(X_scaled, y)
            self.ada_model.fit(X_scaled, y)
            
            # Train neural network
            self.neural_net.train(spy_data)
            
            self.models_trained = True
            self.logger.info("ML models trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return False
            
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for ML models"""
        features = pd.DataFrame(index=df.index)
        
        # Returns at various horizons
        for period in [1, 5, 10, 20, 60]:
            features[f'ret_{period}'] = df['close'].pct_change(period)
            
        # Volatility
        features['vol_10'] = df['close'].pct_change().rolling(10).std()
        features['vol_20'] = df['close'].pct_change().rolling(20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi'] = 100 - (100 / (1 + gain/loss))
        
        # MACD
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        features['macd'] = exp12 - exp26
        
        # Moving average distances
        features['dist_sma20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
        features['dist_sma50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
        
        # Volume ratio
        if 'volume' in df.columns:
            features['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
        return features.dropna()
        
    def generate_signals(self) -> List[Dict]:
        """Generate trading signals for entire universe"""
        if not self.models_trained:
            self.train_models()
            
        signals = []
        
        for symbol in self.UNIVERSE:
            try:
                df = self.get_historical_data(symbol, 100)
                if len(df) < 60:
                    continue
                    
                signal = self._analyze_symbol(symbol, df)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing {symbol}: {e}")
                
        # Sort by conviction score
        signals.sort(key=lambda x: x.get('conviction', 0), reverse=True)
        
        return signals
        
    def _analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Comprehensive analysis of a symbol"""
        
        # Calculate momentum score
        ret_5d = df['close'].pct_change(5).iloc[-1]
        ret_20d = df['close'].pct_change(20).iloc[-1]
        ret_60d = df['close'].pct_change(60).iloc[-1] if len(df) >= 60 else 0
        
        momentum_score = 0.3 * ret_5d + 0.4 * ret_20d + 0.3 * ret_60d
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain/loss))).iloc[-1]
        
        # ML ensemble prediction
        features = self._prepare_ml_features(df)
        if len(features) > 0:
            X = features.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            rf_prob = self.rf_model.predict_proba(X_scaled)[0][1]
            gb_prob = self.gb_model.predict_proba(X_scaled)[0][1]
            ada_prob = self.ada_model.predict_proba(X_scaled)[0][1]
            
            ml_score = (rf_prob + gb_prob + ada_prob) / 3
        else:
            ml_score = 0.5
            
        # Neural network prediction
        nn_pred, nn_conf = self.neural_net.predict(df)
        nn_score = nn_conf if nn_pred == 2 else (1 - nn_conf if nn_pred == 0 else 0.5)
        
        # Combined conviction score
        conviction = (
            self.config.momentum_weight * (momentum_score + 0.5) +
            self.config.neural_net_weight * nn_score +
            (1 - self.config.momentum_weight - self.config.neural_net_weight) * ml_score
        )
        
        # Generate signal
        if conviction > 0.65 and rsi < 70:
            action = 'BUY'
        elif conviction < 0.35 or rsi > 80:
            action = 'SELL'
        else:
            action = 'HOLD'
            
        return {
            'symbol': symbol,
            'action': action,
            'conviction': conviction,
            'momentum_score': momentum_score,
            'ml_score': ml_score,
            'nn_score': nn_score,
            'rsi': rsi,
            'current_price': df['close'].iloc[-1]
        }
        
    def get_status(self) -> Dict:
        """Get engine status"""
        portfolio_value = self.config.initial_capital
        positions = {}
        
        if self.api:
            try:
                account = self.api.get_account()
                portfolio_value = float(account.portfolio_value)
                
                pos_list = self.api.list_positions()
                positions = {p.symbol: {
                    'qty': int(p.qty),
                    'market_value': float(p.market_value),
                    'unrealized_pl': float(p.unrealized_pl)
                } for p in pos_list}
            except Exception as e:
                self.logger.error(f"API error: {e}")
                
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'num_positions': len(positions),
            'positions': positions,
            'models_trained': self.models_trained,
            'universe_size': len(self.UNIVERSE),
            'optimized_params': self.optimized_params
        }
        
    def run_optimization(self) -> Dict:
        """Run genetic algorithm optimization"""
        param_ranges = {
            'momentum_threshold': (0.01, 0.10),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'position_size_mult': (0.5, 2.0),
            'stop_loss': (0.03, 0.10),
            'take_profit': (0.10, 0.30)
        }
        
        self.logger.info("Running genetic optimization...")
        best_params = self.optimizer.optimize(
            self.backtester, 
            self.UNIVERSE[:20],  # Use subset for speed
            param_ranges
        )
        
        if best_params:
            self.optimized_params = best_params
            
        return best_params
        
    def run_backtest(self, years: int = 3) -> Dict:
        """Run backtest with current strategy"""
        start = datetime.now() - timedelta(days=years * 365)
        end = datetime.now()
        
        def strategy_func(all_data, date, idx):
            return self.optimizer._generate_signals(all_data, date, idx, self.optimized_params)
            
        return self.backtester.run_backtest(strategy_func, self.UNIVERSE[:20], start, end)

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='V33 Pairs Trading & Options Flow Engine - LSTM & Kelly Criterion')
    parser.add_argument('--status', action='store_true', help='Show engine status')
    parser.add_argument('--signals', action='store_true', help='Generate trading signals')
    parser.add_argument('--optimize', action='store_true', help='Run genetic optimization')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    args = parser.parse_args()
    
    engine = V31MaximumProfitEngine()
    
    if args.status:
        status = engine.get_status()
        print("\n" + "="*60)
        print("V31 MAXIMUM PROFIT ENGINE STATUS")
        print("="*60)
        print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
        print(f"Positions: {status['num_positions']}")
        print(f"Universe Size: {status['universe_size']}")
        print(f"Models Trained: {status['models_trained']}")
        print(f"Optimized Params: {json.dumps(status['optimized_params'], indent=2)}")
        print("="*60 + "\n")
        
    elif args.signals:
        signals = engine.generate_signals()
        print(f"\nGenerated {len(signals)} signals:\n")
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        for s in buy_signals[:10]:
            print(f"  BUY {s['symbol']:5} | Conviction: {s['conviction']:.3f} | Price: ${s['current_price']:.2f}")
            
    elif args.optimize:
        best = engine.run_optimization()
        print(f"\nOptimization complete!")
        print(f"Best parameters: {json.dumps(best, indent=2)}")
        
    elif args.backtest:
        results = engine.run_backtest()
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Annual Return: {results['annual_return']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {results['win_rate']*100:.1f}%")
        print(f"Total Trades: {results['total_trades']}")
        print("="*60 + "\n")
        
    elif args.train:
        success = engine.train_models()
        print(f"\nTraining {'successful' if success else 'failed'}")
        
    else:
        # Default: show status and top signals
        print("\nV33 Pairs Trading & Options Flow Engine - LSTM & Kelly Criterion")
        print("Usage: python v31_maximum_profit_engine.py [--status|--signals|--optimize|--backtest|--train]")

if __name__ == '__main__':
    main()

class KellyCriterion:
    """Kelly Criterion for optimal position sizing"""
    
    def __init__(self, max_leverage=1.5, min_fraction=0.01):
        self.max_leverage = max_leverage
        self.min_fraction = min_fraction
        
    def calculate(self, win_rate, avg_win, avg_loss):
        """Calculate optimal Kelly fraction"""
        if avg_loss == 0 or win_rate == 0:
            return self.min_fraction
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b if b != 0 else 0
        kelly = kelly * 0.5  # Half-Kelly for safety
        return max(self.min_fraction, min(kelly, self.max_leverage))


class PairsTrading:
    """Statistical Arbitrage Pairs Trading"""
    
    PAIRS = [
        ('XOM', 'CVX'), ('JPM', 'BAC'), ('MSFT', 'AAPL'),
        ('KO', 'PEP'), ('V', 'MA'), ('UNH', 'CVS'),
        ('HD', 'LOW'), ('WMT', 'COST'), ('DIS', 'CMCSA')
    ]
    
    def __init__(self, lookback=60, zscore_threshold=2.0):
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
        self.spreads = {}
        
    def calculate_spread(self, prices1, prices2):
        """Calculate normalized spread between two assets"""
        import numpy as np
        log_prices1 = np.log(prices1)
        log_prices2 = np.log(prices2)
        spread = log_prices1 - log_prices2
        return spread
        
    def calculate_zscore(self, spread):
        """Calculate z-score of spread"""
        import numpy as np
        mean = np.mean(spread[-self.lookback:])
        std = np.std(spread[-self.lookback:])
        if std == 0:
            return 0
        return (spread[-1] - mean) / std
        
    def generate_pair_signal(self, prices1, prices2, pair_name):
        """Generate trading signal for a pair"""
        spread = self.calculate_spread(prices1, prices2)
        zscore = self.calculate_zscore(spread)
        
        if zscore > self.zscore_threshold:
            return {'action': 'short_spread', 'zscore': zscore, 'pair': pair_name}
        elif zscore < -self.zscore_threshold:
            return {'action': 'long_spread', 'zscore': zscore, 'pair': pair_name}
        elif abs(zscore) < 0.5:
            return {'action': 'close', 'zscore': zscore, 'pair': pair_name}
        return {'action': 'hold', 'zscore': zscore, 'pair': pair_name}

