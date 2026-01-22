"""
Anomaly-Aware Transformer
=========================

V2.2 Transformer with integrated anomaly detection for robust predictions.

Architecture:
- Isolation Forest for unsupervised anomaly detection
- Multi-head attention with anomaly-aware masking
- Uncertainty quantification via dropout inference

Key Features:
- Reduces attention on anomalous periods
- Confidence scoring for predictions
- Graceful degradation during market stress
- Attention analysis for interpretability

Research Basis:
- Isolation Forest achieves robust outlier detection O(n log n)
- Attention masking improves transformer robustness
- Uncertainty quantification prevents overconfident trades
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)

# Try to import sklearn for Isolation Forest
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - using simplified anomaly detection")

# Try to import PyTorch for Transformer
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using simplified transformer fallback")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TransformerConfig:
    """Configuration for Anomaly-Aware Transformer."""
    
    # Model dimensions
    input_dim: int = 32          # Input feature dimension
    model_dim: int = 64          # Internal model dimension
    output_dim: int = 1          # Output dimension (signal)
    num_heads: int = 4           # Attention heads
    num_layers: int = 2          # Transformer layers
    ff_dim: int = 128            # Feedforward dimension
    
    # Sequence parameters
    seq_length: int = 20         # Input sequence length
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Anomaly detection
    contamination: float = 0.05  # Expected anomaly fraction
    anomaly_penalty: float = 0.5 # Reduce attention on anomalies
    n_estimators: int = 100      # Isolation Forest trees
    
    # Uncertainty estimation
    mc_dropout_samples: int = 10 # Monte Carlo dropout samples
    confidence_threshold: float = 0.3  # Min confidence to trade
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# ISOLATION FOREST DETECTOR
# =============================================================================

class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector.
    
    Detects unusual market conditions that may compromise
    model predictions. Fast O(n log n) training and inference.
    
    Usage:
        detector = IsolationForestDetector()
        detector.fit(historical_features)
        is_anomaly, score = detector.detect(current_features)
    """
    
    def __init__(self,
                 contamination: float = 0.05,
                 n_estimators: int = 100,
                 max_samples: Union[int, str] = "auto",
                 random_state: int = 42):
        """
        Initialize detector.
        
        Args:
            contamination: Expected fraction of anomalies
            n_estimators: Number of isolation trees
            max_samples: Samples per tree ("auto" or int)
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=random_state,
                n_jobs=-1,
            )
            self.scaler = StandardScaler()
        else:
            self.model = None
            self.scaler = None
            
        self.is_fitted = False
        self.feature_means = None
        self.feature_stds = None
        
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit detector on historical data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for chaining
        """
        if SKLEARN_AVAILABLE:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
        else:
            # Fallback: store statistics for z-score detection
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-8
            
        self.is_fitted = True
        logger.info(f"IsolationForest fitted on {len(X)} samples")
        return self
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
            is_anomaly: Boolean array
            anomaly_score: Score in [0, 1], higher = more anomalous
        """
        if not self.is_fitted:
            return np.zeros(len(X), dtype=bool), np.zeros(len(X))
            
        if SKLEARN_AVAILABLE:
            X_scaled = self.scaler.transform(X)
            
            # -1 = anomaly, 1 = normal
            predictions = self.model.predict(X_scaled)
            is_anomaly = predictions == -1
            
            # Anomaly score (higher = more anomalous)
            # decision_function returns negative for anomalies
            raw_scores = self.model.decision_function(X_scaled)
            anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
        else:
            # Fallback: z-score based detection
            z_scores = np.abs((X - self.feature_means) / self.feature_stds)
            max_z = np.max(z_scores, axis=1)
            
            is_anomaly = max_z > 3.0  # 3 sigma threshold
            anomaly_scores = np.clip(max_z / 5.0, 0, 1)
            
        return is_anomaly, anomaly_scores
    
    def get_anomaly_mask(self, X: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """
        Get binary mask for attention weighting.
        
        Args:
            X: Feature matrix
            threshold: Score threshold for anomaly
            
        Returns:
            Mask array (1 = normal, penalty = anomaly)
        """
        _, scores = self.detect(X)
        mask = np.where(scores > threshold, 0.5, 1.0)  # Reduce attention on anomalies
        return mask


# =============================================================================
# ATTENTION ANALYZER
# =============================================================================

class AttentionAnalyzer:
    """
    Analyzes transformer attention patterns for interpretability.
    
    Provides insights into:
    - Which time steps receive most attention
    - Feature importance via attention gradients
    - Anomaly impact on attention distribution
    """
    
    def __init__(self):
        self.attention_history: List[np.ndarray] = []
        self.max_history = 1000
        
    def record(self, attention_weights: np.ndarray, 
               anomaly_mask: Optional[np.ndarray] = None,
               metadata: Optional[Dict] = None):
        """
        Record attention weights for analysis.
        
        Args:
            attention_weights: (batch, heads, seq, seq) attention matrix
            anomaly_mask: Optional anomaly mask
            metadata: Optional additional info
        """
        record = {
            "attention": attention_weights.mean(axis=(0, 1)),  # Average over batch/heads
            "timestamp": datetime.now().isoformat(),
        }
        
        if anomaly_mask is not None:
            record["anomaly_mask"] = anomaly_mask
        if metadata is not None:
            record.update(metadata)
            
        self.attention_history.append(record)
        
        # Limit history size
        if len(self.attention_history) > self.max_history:
            self.attention_history = self.attention_history[-self.max_history:]
            
    def get_temporal_importance(self, last_n: int = 100) -> np.ndarray:
        """
        Get average attention to each time step.
        
        Args:
            last_n: Number of recent records to analyze
            
        Returns:
            Importance scores for each time step
        """
        if not self.attention_history:
            return np.array([])
            
        recent = self.attention_history[-last_n:]
        attentions = [r["attention"] for r in recent]
        
        # Average attention to each position (last row = prediction step)
        mean_attention = np.mean(attentions, axis=0)
        return mean_attention[-1] if len(mean_attention.shape) > 1 else mean_attention
    
    def get_attention_entropy(self, last_n: int = 100) -> float:
        """
        Compute attention entropy (higher = more distributed).
        
        Args:
            last_n: Number of recent records
            
        Returns:
            Average entropy
        """
        if not self.attention_history:
            return 0.0
            
        recent = self.attention_history[-last_n:]
        entropies = []
        
        for record in recent:
            attn = record["attention"]
            if len(attn.shape) > 1:
                attn = attn[-1]  # Last row (prediction attention)
            attn = attn / (attn.sum() + 1e-8)
            entropy = -np.sum(attn * np.log(attn + 1e-8))
            entropies.append(entropy)
            
        return float(np.mean(entropies))
    
    def get_anomaly_attention_correlation(self, last_n: int = 100) -> float:
        """
        Compute correlation between anomaly mask and attention.
        
        Negative correlation indicates attention is properly reduced on anomalies.
        """
        if not self.attention_history:
            return 0.0
            
        recent = [r for r in self.attention_history[-last_n:] if "anomaly_mask" in r]
        if len(recent) < 10:
            return 0.0
            
        correlations = []
        for record in recent:
            attn = record["attention"]
            mask = record["anomaly_mask"]
            
            if len(attn.shape) > 1:
                attn = attn[-1]
            if len(mask.shape) > 1:
                mask = mask[0]
                
            if len(attn) == len(mask):
                corr = np.corrcoef(attn, 1 - mask)[0, 1]  # Invert mask (1 = anomaly)
                if not np.isnan(corr):
                    correlations.append(corr)
                    
        return float(np.mean(correlations)) if correlations else 0.0


# =============================================================================
# TRANSFORMER COMPONENTS (PyTorch)
# =============================================================================

if TORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequences."""
        
        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            
            self.register_buffer('pe', pe)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    
    class AnomalyAwareAttention(nn.Module):
        """
        Multi-head attention with anomaly-aware masking.
        
        Reduces attention to anomalous time steps based on
        Isolation Forest scores.
        """
        
        def __init__(self, 
                     d_model: int,
                     num_heads: int,
                     dropout: float = 0.1,
                     anomaly_penalty: float = 0.5):
            super().__init__()
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.anomaly_penalty = anomaly_penalty
            
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = np.sqrt(self.head_dim)
            
            # Store attention weights for analysis
            self.last_attention_weights = None
            
        def forward(self, 
                    x: torch.Tensor,
                    anomaly_scores: Optional[torch.Tensor] = None,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Forward pass with optional anomaly masking.
            
            Args:
                x: Input tensor (batch, seq, d_model)
                anomaly_scores: Optional anomaly scores (batch, seq)
                mask: Optional attention mask
                
            Returns:
                Output tensor (batch, seq, d_model)
            """
            batch_size, seq_len, _ = x.shape
            
            # Project to Q, K, V
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Transpose for attention: (batch, heads, seq, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            # Apply anomaly penalty
            if anomaly_scores is not None:
                # anomaly_scores: (batch, seq) -> (batch, 1, 1, seq) for broadcasting
                anomaly_mask = anomaly_scores.unsqueeze(1).unsqueeze(2)
                
                # Penalize attention to anomalous positions
                penalty = self.anomaly_penalty * anomaly_mask
                attn_scores = attn_scores - penalty
                
            # Apply causal/padding mask if provided
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                
            # Softmax and dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Store for analysis
            self.last_attention_weights = attn_weights.detach().cpu().numpy()
            
            # Apply attention to values
            output = torch.matmul(attn_weights, v)
            
            # Reshape and project
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.out_proj(output)
            
            return output
    
    
    class TransformerBlock(nn.Module):
        """Single transformer block with anomaly-aware attention."""
        
        def __init__(self, config: TransformerConfig):
            super().__init__()
            
            self.attention = AnomalyAwareAttention(
                config.model_dim,
                config.num_heads,
                config.attention_dropout,
                config.anomaly_penalty,
            )
            
            self.norm1 = nn.LayerNorm(config.model_dim)
            self.norm2 = nn.LayerNorm(config.model_dim)
            
            self.ff = nn.Sequential(
                nn.Linear(config.model_dim, config.ff_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ff_dim, config.model_dim),
                nn.Dropout(config.dropout),
            )
            
        def forward(self, 
                    x: torch.Tensor,
                    anomaly_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Self-attention with residual
            attn_out = self.attention(x, anomaly_scores)
            x = self.norm1(x + attn_out)
            
            # Feedforward with residual
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            
            return x


# =============================================================================
# ANOMALY-AWARE TRANSFORMER
# =============================================================================

class AnomalyAwareTransformer:
    """
    Transformer with integrated anomaly detection for market prediction.
    
    Combines Isolation Forest anomaly detection with transformer
    attention to create robust predictions that gracefully degrade
    during unusual market conditions.
    
    Features:
    - Pre-screening with Isolation Forest
    - Anomaly-weighted attention
    - Monte Carlo dropout for uncertainty
    - Attention analysis for interpretability
    
    Usage:
        model = AnomalyAwareTransformer(TransformerConfig())
        model.fit(X_train, y_train)
        predictions, confidence = model.predict_with_confidence(X_test)
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        """
        Initialize transformer with anomaly detection.
        
        Args:
            config: Model configuration
        """
        self.config = config or TransformerConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        # Anomaly detector
        self.anomaly_detector = IsolationForestDetector(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
        )
        
        # Attention analyzer
        self.attention_analyzer = AttentionAnalyzer()
        
        # Initialize model if PyTorch available
        if TORCH_AVAILABLE:
            self._init_model()
        else:
            self._init_fallback()
            
        # Training state
        self.is_fitted = False
        self.training_history = []
        
        logger.info(f"AnomalyAwareTransformer initialized on {self.device}")
        
    def _init_model(self):
        """Initialize PyTorch transformer model."""
        
        class TransformerModel(nn.Module):
            def __init__(self, config: TransformerConfig):
                super().__init__()
                
                self.input_proj = nn.Linear(config.input_dim, config.model_dim)
                self.pos_encoder = PositionalEncoding(config.model_dim, config.seq_length)
                
                self.blocks = nn.ModuleList([
                    TransformerBlock(config) for _ in range(config.num_layers)
                ])
                
                self.output_proj = nn.Sequential(
                    nn.Linear(config.model_dim, config.model_dim // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.model_dim // 2, config.output_dim),
                    nn.Tanh(),  # Bound output to [-1, 1] for signal
                )
                
            def forward(self, 
                        x: torch.Tensor,
                        anomaly_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
                # Project input
                x = self.input_proj(x)
                x = self.pos_encoder(x)
                
                # Transformer blocks
                for block in self.blocks:
                    x = block(x, anomaly_scores)
                    
                # Output projection (use last sequence position)
                x = x[:, -1, :]
                return self.output_proj(x)
                
        self.model = TransformerModel(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.criterion = nn.MSELoss()
        
    def _init_fallback(self):
        """Initialize numpy-based fallback (simplified)."""
        logger.warning("Using simplified numpy fallback for transformer")
        self.weights = np.random.randn(self.config.input_dim, self.config.output_dim) * 0.01
        
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train transformer on data.
        
        Args:
            X: Training sequences (n_samples, seq_length, input_dim)
            y: Training targets (n_samples, output_dim)
            X_val: Optional validation sequences
            y_val: Optional validation targets
            
        Returns:
            Training history
        """
        logger.info(f"Training transformer on {len(X)} samples")
        
        # Fit anomaly detector on flattened features
        X_flat = X.reshape(-1, X.shape[-1])
        self.anomaly_detector.fit(X_flat)
        
        if not TORCH_AVAILABLE:
            return self._fit_fallback(X, y)
            
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Get anomaly scores for batch
                batch_flat = batch_X.view(-1, self.config.input_dim).cpu().numpy()
                _, anomaly_scores = self.anomaly_detector.detect(batch_flat)
                anomaly_scores = anomaly_scores.reshape(batch_X.shape[0], batch_X.shape[1])
                anomaly_tensor = torch.FloatTensor(anomaly_scores).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_X, anomaly_tensor)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            epoch_loss /= len(dataloader)
            
            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
            })
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f if val_loss else 'N/A'}")
                
        self.is_fitted = True
        
        return {
            "epochs": len(self.training_history),
            "final_train_loss": self.training_history[-1]["train_loss"],
            "final_val_loss": self.training_history[-1].get("val_loss"),
            "best_val_loss": best_val_loss,
        }
    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Simplified training for numpy fallback."""
        # Simple linear regression
        X_flat = X[:, -1, :]  # Use last sequence element
        self.weights = np.linalg.lstsq(X_flat, y, rcond=None)[0]
        self.is_fitted = True
        return {"method": "fallback"}
    
    def _validate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute validation loss."""
        if not TORCH_AVAILABLE:
            return 0.0
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)
            
        return loss.item()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X: Input sequences (n_samples, seq_length, input_dim)
            
        Returns:
            Predictions (n_samples, output_dim)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, returning zeros")
            return np.zeros((len(X), self.config.output_dim))
            
        if not TORCH_AVAILABLE:
            return np.dot(X[:, -1, :], self.weights)
            
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Get anomaly scores
            X_flat = X.reshape(-1, self.config.input_dim)
            _, anomaly_scores = self.anomaly_detector.detect(X_flat)
            anomaly_scores = anomaly_scores.reshape(X.shape[0], X.shape[1])
            anomaly_tensor = torch.FloatTensor(anomaly_scores).to(self.device)
            
            predictions = self.model(X_tensor, anomaly_tensor)
            
            # Record attention for analysis
            for block in self.model.blocks:
                if hasattr(block.attention, 'last_attention_weights'):
                    self.attention_analyzer.record(
                        block.attention.last_attention_weights,
                        anomaly_scores,
                    )
                    
        return predictions.cpu().numpy()
    
    def predict_with_confidence(self, 
                                X: np.ndarray,
                                n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty quantification.
        
        Uses Monte Carlo dropout to estimate prediction uncertainty.
        
        Args:
            X: Input sequences (n_samples, seq_length, input_dim)
            n_samples: Number of MC samples (default: config.mc_dropout_samples)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            return np.zeros((len(X), self.config.output_dim)), np.zeros(len(X))
            
        if not TORCH_AVAILABLE:
            preds = self.predict(X)
            return preds, np.ones(len(X)) * 0.5
            
        n_samples = n_samples or self.config.mc_dropout_samples
        
        # Enable dropout for MC sampling
        self.model.train()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get anomaly scores
        X_flat = X.reshape(-1, self.config.input_dim)
        _, anomaly_scores = self.anomaly_detector.detect(X_flat)
        anomaly_scores = anomaly_scores.reshape(X.shape[0], X.shape[1])
        anomaly_tensor = torch.FloatTensor(anomaly_scores).to(self.device)
        
        # MC dropout sampling
        predictions_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_tensor, anomaly_tensor)
                predictions_list.append(pred.cpu().numpy())
                
        predictions = np.array(predictions_list)
        
        # Mean prediction and confidence
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence: inverse of normalized std (higher std = lower confidence)
        max_std = np.max(std_pred) + 1e-8
        confidence = 1 - (std_pred.flatten() / max_std)
        
        # Also factor in anomaly score (reduce confidence for anomalous inputs)
        mean_anomaly = np.mean(anomaly_scores, axis=1)
        confidence = confidence * (1 - 0.5 * mean_anomaly)
        
        self.model.eval()
        
        return mean_pred, confidence
    
    def get_signal_with_threshold(self,
                                  X: np.ndarray,
                                  threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trading signals with confidence threshold filtering.
        
        Args:
            X: Input sequences
            threshold: Confidence threshold (default: config.confidence_threshold)
            
        Returns:
            Tuple of (filtered_signals, confidence_scores)
        """
        threshold = threshold or self.config.confidence_threshold
        
        predictions, confidence = self.predict_with_confidence(X)
        
        # Zero out low-confidence predictions
        filtered = predictions.copy()
        low_conf_mask = confidence < threshold
        filtered[low_conf_mask] = 0.0
        
        return filtered, confidence
    
    def analyze_attention(self) -> Dict[str, Any]:
        """
        Get attention analysis summary.
        
        Returns:
            Dictionary with attention statistics
        """
        return {
            "temporal_importance": self.attention_analyzer.get_temporal_importance().tolist()
                if len(self.attention_analyzer.get_temporal_importance()) > 0 else [],
            "attention_entropy": self.attention_analyzer.get_attention_entropy(),
            "anomaly_attention_correlation": self.attention_analyzer.get_anomaly_attention_correlation(),
            "num_records": len(self.attention_analyzer.attention_history),
        }
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        
        if TORCH_AVAILABLE:
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config.to_dict(),
                'training_history': self.training_history,
            }, path)
            
        # Save anomaly detector separately
        import pickle
        with open(path.with_suffix('.anomaly.pkl'), 'wb') as f:
            pickle.dump({
                'is_fitted': self.anomaly_detector.is_fitted,
                'feature_means': self.anomaly_detector.feature_means,
                'feature_stds': self.anomaly_detector.feature_stds,
            }, f)
            
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        if TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.training_history = checkpoint.get('training_history', [])
            
        # Load anomaly detector
        import pickle
        anomaly_path = path.with_suffix('.anomaly.pkl')
        if anomaly_path.exists():
            with open(anomaly_path, 'rb') as f:
                data = pickle.load(f)
                self.anomaly_detector.is_fitted = data.get('is_fitted', False)
                self.anomaly_detector.feature_means = data.get('feature_means')
                self.anomaly_detector.feature_stds = data.get('feature_stds')
                
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary for logging."""
        return {
            "config": self.config.to_dict(),
            "is_fitted": self.is_fitted,
            "training_epochs": len(self.training_history),
            "anomaly_detector_fitted": self.anomaly_detector.is_fitted,
            "device": str(self.device) if self.device else "cpu",
            "torch_available": TORCH_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
        }
