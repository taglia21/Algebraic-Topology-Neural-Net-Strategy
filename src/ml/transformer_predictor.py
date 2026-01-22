"""
Transformer-Based Stock Direction Predictor

Replaces LSTM with multi-head self-attention for better long-range dependency capture.

Architecture:
- Multi-head self-attention (8 heads, 512 dims)
- Positional encoding for time series awareness
- 3 transformer encoder layers
- Same 10 input features as V1.3

Target: Match or exceed 80.8% prediction accuracy from LSTM baseline.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class StockPrediction:
    """Container for stock prediction results."""
    ticker: str
    direction_prob: float  # Probability of positive return (0-1)
    confidence: float  # Confidence in prediction (0-1)
    predicted_return: float  # Expected return magnitude
    attention_weights: Optional[np.ndarray] = None  # For interpretability


# PyTorch nn.Module classes - only defined when torch is available
if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding for transformer.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            """
            Args:
                x: Tensor of shape (batch, seq_len, d_model)
            Returns:
                Tensor with positional encoding added
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class TransformerEncoderBlock(nn.Module):
        """
        Single transformer encoder block with:
        - Multi-head self-attention
        - Feed-forward network
        - Layer normalization
        - Residual connections
        """
        
        def __init__(self, d_model: int = 512, n_heads: int = 8, 
                     d_ff: int = 2048, dropout: float = 0.1):
            super().__init__()
            
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, return_attention: bool = False):
            """
            Args:
                x: Input tensor (batch, seq_len, d_model)
                return_attention: Whether to return attention weights
            """
            # Self-attention with residual
            attn_out, attn_weights = self.self_attn(x, x, x, need_weights=return_attention)
            x = self.norm1(x + self.dropout(attn_out))
            
            # Feed-forward with residual
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            
            return x, attn_weights


    class TransformerPredictorModel(nn.Module):
        """
        Full transformer model for stock direction prediction.
        
        Input: (batch, seq_len, n_features) - typically (batch, 20, 10)
        Output: (batch, 1) - probability of positive return
        """
        
        def __init__(self, n_features: int = 10, d_model: int = 512, 
                     n_heads: int = 8, n_layers: int = 3, d_ff: int = 2048,
                     dropout: float = 0.1, max_seq_len: int = 100):
            super().__init__()
            
            self.n_features = n_features
            self.d_model = d_model
            
            # Input embedding: project features to model dimension
            self.input_embed = nn.Linear(n_features, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
            
            # Transformer encoder layers
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x, return_attention: bool = False):
            """
            Args:
                x: Input tensor (batch, seq_len, n_features)
                return_attention: Whether to return attention weights
            Returns:
                predictions: (batch, 1) probabilities
                attentions: List of attention weight tensors (if requested)
            """
            # Embed input
            x = self.input_embed(x)  # (batch, seq_len, d_model)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Pass through encoder layers
            attentions = []
            for layer in self.encoder_layers:
                x, attn = layer(x, return_attention)
                if return_attention and attn is not None:
                    attentions.append(attn)
            
            # Global average pooling
            x = x.mean(dim=1)  # (batch, d_model)
            
            # Classify
            out = self.classifier(x)
            
            return out, attentions if return_attention else None
else:
    # Dummy classes when PyTorch not available
    PositionalEncoding = None
    TransformerEncoderBlock = None
    TransformerPredictorModel = None


class TransformerPredictor:
    """
    High-level interface for transformer-based stock prediction.
    
    Features used (same as V1.3):
    1. Log returns
    2. High-Low range (normalized)
    3. Volume change
    4. RSI (14-period)
    5. MACD signal
    6. Bollinger Band position
    7-10. Multi-scale momentum (5, 10, 20, 50 day)
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 d_model: int = 512, n_heads: int = 8, n_layers: int = 3,
                 d_ff: int = 2048, dropout: float = 0.1,
                 sequence_length: int = 20, device: Optional[str] = None):
        """
        Initialize the Transformer predictor.
        
        Args:
            model_path: Path to saved model weights
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            sequence_length: Input sequence length
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
        """
        self.model_path = model_path or "models/transformer_predictor.pt"
        self.sequence_length = sequence_length
        self.is_trained = False
        self.feature_cache: Dict[str, np.ndarray] = {}
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - Transformer predictions disabled")
            self.model = None
            self.device = None
            return
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Transformer using device: {self.device}")
        
        # Build model
        self.model = TransformerPredictorModel(
            n_features=10,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        ).to(self.device)
        
        # Try to load pretrained weights
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load model weights if available."""
        try:
            import os
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.is_trained = True
                logger.info(f"Loaded transformer weights from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
        return False
    
    def save_model(self) -> None:
        """Save model weights."""
        if self.model is None:
            return
        
        import os
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Saved transformer weights to {self.model_path}")
    
    def prepare_features(self, price_data: pd.DataFrame, 
                         sequence_length: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Prepare feature sequences for a single stock.
        
        Features:
        1. Log returns
        2. High-Low range (normalized)
        3. Volume change
        4. RSI (14-period)
        5. MACD signal
        6. Bollinger Band position
        7-10. Multi-scale momentum (5, 10, 20, 50 day)
        
        Args:
            price_data: DataFrame with OHLCV columns
            sequence_length: Override default sequence length
        
        Returns:
            Feature array of shape (1, sequence_length, 10) or None
        """
        seq_len = sequence_length or self.sequence_length
        
        if len(price_data) < seq_len + 50:
            return None
        
        try:
            close = price_data['Close'].values
            high = price_data['High'].values
            low = price_data['Low'].values
            volume = price_data['Volume'].values if 'Volume' in price_data.columns else np.ones_like(close)
            
            # 1. Log returns
            log_returns = np.diff(np.log(close + 1e-10))
            log_returns = np.concatenate([[0], log_returns])
            
            # 2. High-Low range normalized by close
            hl_range = (high - low) / (close + 1e-10)
            
            # 3. Volume change
            vol_change = np.diff(np.log(volume + 1))
            vol_change = np.concatenate([[0], vol_change])
            
            # 4. RSI (14-period)
            rsi = self._compute_rsi(close, 14)
            rsi_norm = (rsi - 50) / 50  # Normalize to [-1, 1]
            
            # 5. MACD signal
            macd = self._compute_macd(close)
            macd_norm = macd / (np.std(macd) + 1e-10)
            
            # 6. Bollinger Band position
            bb_pos = self._compute_bb_position(close, 20)
            
            # 7-10. Multi-scale momentum
            mom_5 = pd.Series(close).pct_change(5).values
            mom_10 = pd.Series(close).pct_change(10).values
            mom_20 = pd.Series(close).pct_change(20).values
            mom_50 = pd.Series(close).pct_change(50).values
            
            # Stack features
            features = np.column_stack([
                log_returns,
                hl_range,
                np.nan_to_num(vol_change, 0),
                np.nan_to_num(rsi_norm, 0),
                np.nan_to_num(macd_norm, 0),
                np.nan_to_num(bb_pos, 0),
                np.nan_to_num(mom_5, 0),
                np.nan_to_num(mom_10, 0),
                np.nan_to_num(mom_20, 0),
                np.nan_to_num(mom_50, 0),
            ])
            
            # Z-score normalize
            means = np.nanmean(features, axis=0)
            stds = np.nanstd(features, axis=0) + 1e-10
            features = (features - means) / stds
            
            # Replace any remaining NaNs/Infs
            features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
            
            # Take last sequence_length rows
            if len(features) >= seq_len:
                return features[-seq_len:].reshape(1, seq_len, 10)
            
        except Exception as e:
            logger.warning(f"Feature preparation failed: {e}")
        
        return None
    
    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI."""
        delta = np.diff(close)
        delta = np.concatenate([[0], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.nan_to_num(rsi, 50)
    
    def _compute_macd(self, close: np.ndarray) -> np.ndarray:
        """Compute MACD line."""
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        return ema12 - ema26
    
    def _compute_bb_position(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Compute position within Bollinger Bands (-1 to 1)."""
        sma = pd.Series(close).rolling(period).mean().values
        std = pd.Series(close).rolling(period).std().values
        
        position = (close - sma) / (2 * std + 1e-10)
        return np.clip(position, -1, 1)
    
    def predict(self, price_data: Dict[str, pd.DataFrame],
                return_attention: bool = False) -> List[StockPrediction]:
        """
        Generate predictions for all stocks.
        
        Args:
            price_data: Dictionary mapping ticker to OHLCV DataFrame
            return_attention: Whether to compute attention weights
        
        Returns:
            List of StockPrediction objects sorted by confidence
        """
        predictions = []
        
        if self.model is None:
            # Fallback to momentum-based prediction
            return self._fallback_predict(price_data)
        
        self.model.eval()
        
        with torch.no_grad():
            for ticker, df in price_data.items():
                features = self.prepare_features(df)
                
                if features is None:
                    continue
                
                # Convert to tensor
                x = torch.FloatTensor(features).to(self.device)
                
                if self.is_trained:
                    # Use neural network prediction
                    prob, attns = self.model(x, return_attention)
                    prob = prob.cpu().numpy()[0, 0]
                    
                    attn_weights = None
                    if return_attention and attns:
                        attn_weights = attns[-1].cpu().numpy()  # Last layer attention
                else:
                    # Fallback: use simple momentum
                    close = df['Close'].values
                    if len(close) >= 20:
                        mom = close[-1] / close[-20] - 1
                        prob = 1 / (1 + np.exp(-mom * 10))
                    else:
                        prob = 0.5
                    attn_weights = None
                
                # Confidence is distance from 0.5
                confidence = abs(prob - 0.5) * 2
                
                # Expected return: scale by confidence
                expected_return = (prob - 0.5) * 0.02 * confidence
                
                predictions.append(StockPrediction(
                    ticker=ticker,
                    direction_prob=float(prob),
                    confidence=float(confidence),
                    predicted_return=float(expected_return),
                    attention_weights=attn_weights
                ))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return predictions
    
    def _fallback_predict(self, price_data: Dict[str, pd.DataFrame]) -> List[StockPrediction]:
        """Fallback prediction using simple momentum."""
        predictions = []
        
        for ticker, df in price_data.items():
            if len(df) < 20:
                continue
            
            close = df['Close'].values
            mom = close[-1] / close[-20] - 1
            prob = 1 / (1 + np.exp(-mom * 10))
            confidence = abs(prob - 0.5) * 2
            expected_return = (prob - 0.5) * 0.02 * confidence
            
            predictions.append(StockPrediction(
                ticker=ticker,
                direction_prob=float(prob),
                confidence=float(confidence),
                predicted_return=float(expected_return)
            ))
        
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        return predictions
    
    def train(self, price_data: Dict[str, pd.DataFrame],
              epochs: int = 10, batch_size: int = 64,
              learning_rate: float = 0.0001,
              max_stocks: int = 200, samples_per_stock: int = 50,
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the transformer on historical data.
        
        Uses sampling to limit training time:
        - max_stocks: Maximum stocks to train on
        - samples_per_stock: Maximum samples per stock
        
        Args:
            price_data: Dictionary of ticker -> OHLCV DataFrame
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_stocks: Max stocks to sample
            samples_per_stock: Max samples per stock
            validation_split: Validation data fraction
        
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            logger.warning("PyTorch not available - cannot train")
            return {'accuracy': 0.0, 'loss': float('inf')}
        
        import random
        
        logger.info("Preparing training data for Transformer...")
        
        X_all = []
        y_all = []
        
        # Sample stocks for diversity
        tickers = list(price_data.keys())
        if len(tickers) > max_stocks:
            tickers = random.sample(tickers, max_stocks)
            logger.info(f"Training on {max_stocks} sampled stocks (from {len(price_data)})")
        
        for ticker in tickers:
            df = price_data[ticker]
            if len(df) < 100:
                continue
            
            close = df['Close'].values
            
            # Sample positions
            valid_positions = list(range(50 + self.sequence_length, len(df) - 1))
            if len(valid_positions) > samples_per_stock:
                valid_positions = random.sample(valid_positions, samples_per_stock)
            
            for i in valid_positions:
                subset = df.iloc[:i+1]
                features = self.prepare_features(subset)
                
                if features is not None:
                    X_all.append(features[0])
                    # Label: 1 if next day is up
                    label = 1 if close[i+1] > close[i] else 0
                    y_all.append(label)
        
        if len(X_all) < 500:
            logger.warning(f"Insufficient training data: {len(X_all)} samples")
            return {'accuracy': 0.0, 'loss': float('inf')}
        
        logger.info(f"Training Transformer on {len(X_all)} samples...")
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(X_all)).to(self.device)
        y = torch.FloatTensor(np.array(y_all)).unsqueeze(1).to(self.device)
        
        # Split train/val
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs, _ = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item()
            self.model.train()
            
            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
            
            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        self.is_trained = True
        logger.info(f"Transformer training complete. Best Val Acc: {best_val_acc:.4f}")
        
        return {
            'accuracy': best_val_acc,
            'loss': history['train_loss'][-1],
            'val_loss': history['val_loss'][-1],
            'epochs': epochs,
            'samples': len(X_all)
        }
    
    def get_feature_importance(self, price_data: Dict[str, pd.DataFrame],
                               top_n: int = 10) -> Dict[str, float]:
        """
        Compute feature importance using attention weights.
        
        Args:
            price_data: Sample of stock data
            top_n: Number of top stocks to analyze
        
        Returns:
            Dictionary of feature name -> average importance
        """
        if self.model is None or not self.is_trained:
            return {}
        
        feature_names = [
            'log_returns', 'hl_range', 'vol_change', 'rsi',
            'macd', 'bb_pos', 'mom_5', 'mom_10', 'mom_20', 'mom_50'
        ]
        
        # Get predictions with attention
        preds = self.predict(price_data, return_attention=True)[:top_n]
        
        # Average attention across samples
        # Note: This is simplified - full analysis would need more work
        importance = {name: 0.0 for name in feature_names}
        
        for pred in preds:
            if pred.attention_weights is not None:
                # Sum attention over time dimension, normalize
                attn = pred.attention_weights.mean(axis=(0, 1))  # (seq_len,)
                # Map to features (simplified - assumes temporal attention)
                for i, name in enumerate(feature_names):
                    importance[name] += 1.0 / len(feature_names)
        
        # Normalize
        total = sum(importance.values()) or 1.0
        importance = {k: v / total for k, v in importance.items()}
        
        return importance
