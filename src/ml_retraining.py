import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from sklearn.model_selection import train_test_split
from tensorflow import keras
import alpaca_trade_api as tradeapi
import os

logger = logging.getLogger(__name__)

class MLRetrainingScheduler:
    def __init__(self, api):
        self.api = api
        self.est = pytz.timezone('US/Eastern')
        self.model_path = 'models/neural_net.h5'
        
    def fetch_training_data(self, symbols, days=30):
        """Fetch historical data for training"""
        end = datetime.now()
        start = end - timedelta(days=days)
        
        all_data = []
        for symbol in symbols:
            try:
                bars = self.api.get_bars(
                    symbol,
                    '1Hour',
                    start=start.isoformat(),
                    end=end.isoformat()
                ).df
                bars['symbol'] = symbol
                all_data.append(bars)
            except Exception as e:
                logger.error(f'Error fetching {symbol}: {e}')
                
        if all_data:
            return pd.concat(all_data)
        return None
        
    def prepare_features(self, df):
        """Prepare features for neural network"""
        # Technical indicators
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        df['vol'] = df.groupby('symbol')['returns'].rolling(20).std()
        df['sma_20'] = df.groupby('symbol')['close'].rolling(20).mean()
        df['sma_50'] = df.groupby('symbol')['close'].rolling(50).mean()
        df['rsi'] = self.calculate_rsi(df.groupby('symbol')['close'])
        
        # TDA features would go here
        # For now, using standard technical features
        
        feature_cols = ['returns', 'vol', 'sma_20', 'sma_50', 'rsi']
        df = df.dropna(subset=feature_cols)
        
        X = df[feature_cols].values
        # Target: next period return
        y = df.groupby('symbol')['returns'].shift(-1).values
        
        # Remove NaN targets
        mask = ~np.isnan(y)
        return X[mask], y[mask]
        
    def calculate_rsi(self, series, period=14):
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def build_model(self, input_dim):
        """Build neural network model"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
        
    async def retrain_model(self, symbols):
        """Retrain neural network on latest data"""
        logger.info('Starting ML model retraining')
        
        # Fetch data
        df = self.fetch_training_data(symbols)
        if df is None:
            logger.error('No training data available')
            return False
            
        # Prepare features
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build and train model
        model = self.build_model(X.shape[1])
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save(self.model_path)
        
        test_loss = history.history['val_loss'][-1]
        logger.info(f'Model retrained. Validation loss: {test_loss:.6f}')
        return True
        
    async def schedule_retraining(self, symbols):
        """Schedule daily retraining at midnight"""
        while True:
            now = datetime.now(self.est)
            
            # Retrain at midnight EST
            if now.hour == 0 and now.minute < 5:
                await self.retrain_model(symbols)
                await asyncio.sleep(300)  # Wait 5 minutes
            else:
                await asyncio.sleep(60)  # Check every minute
