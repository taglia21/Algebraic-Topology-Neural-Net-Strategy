"""LSTM Neural Network Predictor for market direction classification."""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class NeuralNetPredictor(keras.Model):
    """LSTM-based predictor for next bar direction (up/down)."""

    def __init__(self, sequence_length: int = 20, n_features: int = 6, lstm_units: int = 32):
        """Initialize LSTM predictor with given architecture params."""
        super().__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = layers.LSTM(lstm_units // 2, return_sequences=False)
        self.dense1 = layers.Dense(16, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        """Forward pass: (batch, 20, 6) → (batch, 1)."""
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        x = self.dense1(x)
        return self.output_layer(x)

    def compile_model(self, learning_rate: float = 0.001):
        """Compile with Adam optimizer and binary crossentropy loss."""
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def save_checkpoint(self, filepath: str):
        """Save model weights to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.save_weights(filepath)

    def load_checkpoint(self, filepath: str):
        """Load model weights from file."""
        self.load_weights(filepath)


class DataPreprocessor:
    """Prepare OHLCV + TDA data for LSTM training."""

    def __init__(self, sequence_length: int = 20):
        """Initialize with sequence length for sliding window."""
        self.sequence_length = sequence_length
        self.feature_means = None
        self.feature_stds = None

    def prepare_sequences(self, ohlcv_df, tda_features_df) -> tuple:
        """Create X, y sequences from aligned OHLCV and TDA features."""
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_cols = [c if c in ohlcv_df.columns else c.capitalize() for c in ohlcv_cols]
        
        start_idx = len(ohlcv_df) - len(tda_features_df)
        ohlcv_aligned = ohlcv_df.iloc[start_idx:].reset_index(drop=True)
        
        features = self._normalize_features(ohlcv_aligned, tda_features_df)
        
        X, y = self._create_sliding_windows(features, ohlcv_aligned)
        
        return X, y

    def _normalize_features(self, ohlcv_df, tda_df) -> np.ndarray:
        """Normalize OHLCV and TDA features to zero mean, unit variance."""
        close_col = 'close' if 'close' in ohlcv_df.columns else 'Close'
        open_col = 'open' if 'open' in ohlcv_df.columns else 'Open'
        high_col = 'high' if 'high' in ohlcv_df.columns else 'High'
        low_col = 'low' if 'low' in ohlcv_df.columns else 'Low'
        vol_col = 'volume' if 'volume' in ohlcv_df.columns else 'Volume'
        
        close = ohlcv_df[close_col].values
        returns = np.diff(np.log(close + 1e-10))
        returns = np.concatenate([[0], returns])
        
        hl_range = (ohlcv_df[high_col] - ohlcv_df[low_col]) / close
        oc_range = (ohlcv_df[close_col] - ohlcv_df[open_col]) / close
        vol_norm = np.log1p(ohlcv_df[vol_col].values)
        
        tda_signal = tda_df['persistence_l1'].values
        tda_signal = (tda_signal - tda_signal.mean()) / (tda_signal.std() + 1e-10)
        
        features = np.column_stack([
            returns,
            hl_range.values,
            oc_range.values,
            vol_norm,
            tda_df['persistence_l0'].values,
            tda_signal
        ])
        
        if self.feature_means is None:
            self.feature_means = features.mean(axis=0)
            self.feature_stds = features.std(axis=0) + 1e-10
        
        return (features - self.feature_means) / self.feature_stds

    def _create_sliding_windows(self, features: np.ndarray, ohlcv_df) -> tuple:
        """Create sliding window sequences for LSTM input."""
        close_col = 'close' if 'close' in ohlcv_df.columns else 'Close'
        close = ohlcv_df[close_col].values
        
        X_list = []
        y_list = []
        
        for i in range(self.sequence_length, len(features) - 1):
            X_list.append(features[i - self.sequence_length : i])
            y_list.append(1 if close[i + 1] > close[i] else 0)
        
        return np.array(X_list), np.array(y_list)


def train_model(model: NeuralNetPredictor, X_train: np.ndarray, y_train: np.ndarray,
                epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
    """Train the model with early stopping."""
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0
    )
    
    return history


def test():
    """Test neural network predictor on synthetic data."""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model = NeuralNetPredictor(sequence_length=20, n_features=6, lstm_units=32)
    model.compile_model(learning_rate=0.001)
    
    dummy_input = np.random.randn(1, 20, 6).astype(np.float32)
    output = model(dummy_input)
    
    assert output.shape == (1, 1), f"Output shape mismatch: {output.shape}"
    assert 0 <= float(output[0, 0]) <= 1, f"Output not in [0,1]: {output[0, 0]}"
    
    batch_input = np.random.randn(32, 20, 6).astype(np.float32)
    batch_output = model(batch_input)
    assert batch_output.shape == (32, 1), f"Batch output shape: {batch_output.shape}"
    
    X_train = np.random.randn(100, 20, 6).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
    
    model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=0)
    
    os.makedirs('/workspaces/Algebraic-Topology-Neural-Net-Strategy/results', exist_ok=True)
    checkpoint_path = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/test_weights.weights.h5'
    model.save_checkpoint(checkpoint_path)
    
    model2 = NeuralNetPredictor(sequence_length=20, n_features=6, lstm_units=32)
    model2.compile_model()
    _ = model2(dummy_input)
    model2.load_checkpoint(checkpoint_path)
    
    return True


if __name__ == "__main__":
    success = test()
    if success:
        import sys
        sys.stdout.write("Neural Net Predictor: All tests passed\n")
