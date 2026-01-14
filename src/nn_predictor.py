"""LSTM Neural Network Predictor for market direction classification.

Enhanced with entropy penalty loss to combat output variance compression.
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# ENTROPY PENALTY LOSS FUNCTION
# =============================================================================
# Addresses output variance compression where model outputs cluster at 0.5.
# Binary crossentropy naturally compresses variance; entropy penalty encourages
# the model to make more decisive (extreme) predictions.
# =============================================================================

def entropy_penalty_loss(entropy_weight: float = 0.05):
    """
    Create weighted binary crossentropy with entropy penalty.
    
    The entropy penalty discourages outputs near 0.5 by adding a term that
    is maximized when output = 0.5 and minimized at 0 or 1.
    
    Args:
        entropy_weight: Weight for the entropy penalty term (0.05 = 5%)
    
    Returns:
        Loss function compatible with Keras.
    """
    def weighted_binary_crossentropy(y_true, y_pred):
        # Standard binary crossentropy
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Entropy penalty: penalize predictions near 0.5
        # Entropy = -p*log(p) - (1-p)*log(1-p)
        # This is MAXIMIZED at p=0.5 (~0.693) and MINIMIZED at p=0 or p=1 (0)
        entropy = -(y_pred * tf.math.log(y_pred + epsilon) + 
                    (1 - y_pred) * tf.math.log(1 - y_pred + epsilon))
        
        # By adding entropy (not subtracting), we ENCOURAGE extreme outputs
        # because the optimizer will try to MINIMIZE total loss
        # (lower entropy = more extreme output = lower loss)
        return bce + entropy_weight * entropy
    
    return weighted_binary_crossentropy


class OutputSpreadCallback(keras.callbacks.Callback):
    """
    Custom callback to track output spread during training.
    
    Logs per-epoch statistics about model output distribution to detect
    variance compression issues.
    """
    
    def __init__(self, X_val: np.ndarray, verbose: bool = True):
        super().__init__()
        self.X_val = X_val
        self.verbose = verbose
        self.history = {
            'output_mean': [],
            'output_std': [],
            'pct_above_055': [],
            'pct_below_045': [],
            'pct_extreme': []  # > 0.55 OR < 0.45
        }
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on validation data
        predictions = self.model.predict(self.X_val, verbose=0).flatten()
        
        # Calculate statistics
        output_mean = float(np.mean(predictions))
        output_std = float(np.std(predictions))
        pct_above_055 = float(np.mean(predictions > 0.55) * 100)
        pct_below_045 = float(np.mean(predictions < 0.45) * 100)
        pct_extreme = pct_above_055 + pct_below_045
        
        # Store history
        self.history['output_mean'].append(output_mean)
        self.history['output_std'].append(output_std)
        self.history['pct_above_055'].append(pct_above_055)
        self.history['pct_below_045'].append(pct_below_045)
        self.history['pct_extreme'].append(pct_extreme)
        
        if self.verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: mean={output_mean:.4f}, std={output_std:.4f}, "
                  f"extreme={pct_extreme:.1f}% (>{.55}:{pct_above_055:.1f}%, <{.45}:{pct_below_045:.1f}%)")


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

    def compile_model(self, learning_rate: float = 0.001, use_entropy_penalty: bool = False,
                       entropy_weight: float = 0.05):
        """
        Compile with Adam optimizer.
        
        Args:
            learning_rate: Adam learning rate
            use_entropy_penalty: If True, use entropy penalty loss to spread outputs
            entropy_weight: Weight for entropy penalty (only used if use_entropy_penalty=True)
        """
        if use_entropy_penalty:
            loss_fn = entropy_penalty_loss(entropy_weight)
            print(f"    Using entropy penalty loss (weight={entropy_weight})")
        else:
            loss_fn = 'binary_crossentropy'
            print(f"    Using standard binary crossentropy loss")
        
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
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
    """Prepare OHLCV + TDA data for LSTM training.
    
    V1.2: Supports extended TDA features (10 features) in addition to V1.1 (4 features).
    """

    def __init__(self, sequence_length: int = 20, use_extended_tda: bool = True):
        """Initialize with sequence length for sliding window.
        
        Args:
            sequence_length: Number of time steps per sequence
            use_extended_tda: If True, use all TDA columns dynamically (V1.2)
                            If False, use only persistence_l0, persistence_l1 (V1.1)
        """
        self.sequence_length = sequence_length
        self.feature_means = None
        self.feature_stds = None
        self.use_extended_tda = use_extended_tda

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
        """Normalize OHLCV and TDA features to zero mean, unit variance.
        
        V1.2: Dynamically handles extended TDA features.
        """
        close_col = 'close' if 'close' in ohlcv_df.columns else 'Close'
        open_col = 'open' if 'open' in ohlcv_df.columns else 'Open'
        high_col = 'high' if 'high' in ohlcv_df.columns else 'High'
        low_col = 'low' if 'low' in ohlcv_df.columns else 'Low'
        vol_col = 'volume' if 'volume' in ohlcv_df.columns else 'Volume'
        
        close = ohlcv_df[close_col].values
        returns = np.diff(np.log(close + 1e-10))
        returns = np.concatenate([[0], returns])
        
        hl_range = (ohlcv_df[high_col] - ohlcv_df[low_col]) / close
        
        # Build feature list: OHLCV-derived first, then all TDA features
        feature_list = [
            returns,
            hl_range.values,
        ]
        
        # Add all available TDA columns dynamically (V1.2)
        if self.use_extended_tda:
            # Use all numeric columns from tda_df
            for col in tda_df.columns:
                col_vals = tda_df[col].values.astype(float)
                # Normalize each TDA column individually before stacking
                col_mean = col_vals.mean()
                col_std = col_vals.std() + 1e-10
                normalized_col = (col_vals - col_mean) / col_std
                feature_list.append(normalized_col)
        else:
            # V1.1 compatible: only use persistence_l0 and persistence_l1
            tda_l0 = tda_df['persistence_l0'].values
            tda_l0 = (tda_l0 - tda_l0.mean()) / (tda_l0.std() + 1e-10)
            
            tda_l1 = tda_df['persistence_l1'].values
            tda_l1 = (tda_l1 - tda_l1.mean()) / (tda_l1.std() + 1e-10)
            
            feature_list.extend([tda_l0, tda_l1])
        
        features = np.column_stack(feature_list)
        
        # Global normalization across all features
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
                epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2,
                track_output_spread: bool = False, verbose_spread: bool = True):
    """
    Train the model with early stopping and optional output spread tracking.
    
    Args:
        model: NeuralNetPredictor instance
        X_train: Training features
        y_train: Training labels
        epochs: Maximum epochs
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        track_output_spread: If True, track output distribution per epoch
        verbose_spread: If True, print spread stats every 10 epochs
    
    Returns:
        Tuple of (history, spread_callback) if track_output_spread else history
    """
    # Split validation data for spread tracking
    val_size = int(len(X_train) * validation_split)
    X_val_for_spread = X_train[-val_size:]
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience for entropy penalty exploration
            restore_best_weights=True
        )
    ]
    
    spread_callback = None
    if track_output_spread:
        spread_callback = OutputSpreadCallback(X_val_for_spread, verbose=verbose_spread)
        callbacks.append(spread_callback)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=0
    )
    
    if track_output_spread:
        return history, spread_callback
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
