"""
Temporal Transformer for Time Series Prediction
================================================

V2.3 Enhanced Transformer with advanced temporal modeling for
price/volume/TDA time series prediction.

Architecture:
- Multi-head self-attention with learnable positional encoding
- Seasonal and autocorrelation-aware positional encoding
- Macroeconomic variable integration (VIX, credit spreads, EPU)
- TDA feature fusion layer
- Integration with anomaly_aware_transformer.py

Key Features:
- Captures long-range dependencies in price series
- Models market microstructure via attention patterns
- Integrates macro regime indicators
- Supports variable-length sequences
- Uncertainty quantification via MC dropout

Research Basis:
- Transformers outperform RNNs at 1/3/12-month intervals
- Positional encoding critical for seasonality
- Macro variables improve regime detection

Target Performance:
- MAE improvement > 15% vs baseline models
- Latency < 50ms per prediction batch
"""

from __future__ import annotations

import os
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from pathlib import Path
import math

logger = logging.getLogger(__name__)

# Type checking imports (for static analysis only)
if TYPE_CHECKING:
    import torch
    from torch import Tensor

# Try to import PyTorch at runtime
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class _DummyModule:
        pass
    nn = type('nn', (), {'Module': _DummyModule})()
    torch = None  # type: ignore[assignment]
    logger.warning("PyTorch not available - using simplified transformer fallback")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TemporalTransformerConfig:
    """Configuration for Temporal Transformer."""
    
    # Input dimensions
    price_features: int = 5        # OHLCV
    tda_features: int = 20         # TDA features from V1.3
    macro_features: int = 6        # VIX, credit spread, EPU, inflation, etc.
    total_input_dim: int = 31      # Sum of above
    
    # Model architecture
    model_dim: int = 128           # Internal model dimension
    num_heads: int = 8             # Attention heads
    num_encoder_layers: int = 4    # Encoder layers
    num_decoder_layers: int = 2    # Decoder layers (for autoregressive)
    ff_dim: int = 256              # Feedforward dimension
    
    # Sequence parameters
    max_seq_len: int = 252         # ~1 year of trading days
    forecast_horizon: int = 5     # Predict next 5 days
    
    # Positional encoding
    use_learnable_pos: bool = True
    use_seasonal_pos: bool = True  # Add weekly/monthly seasonality
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_dropout: float = 0.0     # Stochastic depth
    
    # Output
    output_dim: int = 1            # Return prediction
    num_quantiles: int = 5         # For quantile regression
    
    # Uncertainty
    mc_dropout_samples: int = 20
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# FALLBACK IMPLEMENTATION
# =============================================================================

class FallbackTemporalTransformer:
    """Simple fallback without PyTorch."""
    
    def __init__(self, config: TemporalTransformerConfig):
        self.config = config
        
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simple moving average prediction."""
        batch_size = x.shape[0]
        
        # Use last values as prediction
        if x.ndim >= 3:
            pred = x[:, -1, 0:1]  # Last close price
        else:
            pred = np.zeros((batch_size, 1))
            
        return pred, {'method': 'fallback_ma'}


# =============================================================================
# PYTORCH IMPLEMENTATION
# =============================================================================

if TORCH_AVAILABLE:
    
    class LearnablePositionalEncoding(nn.Module):
        """Learnable positional encoding with optional seasonal components."""
        
        def __init__(
            self, 
            d_model: int, 
            max_len: int = 500,
            use_seasonal: bool = True,
            dropout: float = 0.1
        ):
            super().__init__()
            self.d_model = d_model
            self.use_seasonal = use_seasonal
            self.dropout = nn.Dropout(p=dropout)
            
            # Learnable position embeddings
            self.position_embed = nn.Embedding(max_len, d_model)
            
            if use_seasonal:
                # Weekly seasonality (5 trading days)
                self.weekly_embed = nn.Embedding(5, d_model // 4)
                # Monthly seasonality (~21 trading days)
                self.monthly_embed = nn.Embedding(21, d_model // 4)
                
                # Projection to combine
                self.seasonal_proj = nn.Linear(d_model // 2, d_model)
            
            # Sinusoidal base (for relative positions)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe_sinusoidal', pe.unsqueeze(0))
            
        def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Add positional encoding to input.
            
            Args:
                x: [batch, seq_len, d_model]
                positions: Optional position indices
                
            Returns:
                x with positional encoding added
            """
            seq_len = x.size(1)
            
            if positions is None:
                positions = torch.arange(seq_len, device=x.device)
            
            # Learnable positional embedding
            pos_embed = self.position_embed(positions)  # [seq_len, d_model]
            
            # Add sinusoidal for relative position awareness
            x = x + pos_embed + 0.1 * self.pe_sinusoidal[:, :seq_len, :]
            
            if self.use_seasonal:
                # Weekly position (day of week)
                weekly_pos = positions % 5
                weekly = self.weekly_embed(weekly_pos)
                
                # Monthly position (day of month)
                monthly_pos = positions % 21
                monthly = self.monthly_embed(monthly_pos)
                
                # Combine seasonal
                seasonal = torch.cat([weekly, monthly], dim=-1)
                seasonal = self.seasonal_proj(seasonal)
                
                x = x + 0.1 * seasonal
            
            return self.dropout(x)


    class MacroEncoder(nn.Module):
        """
        Encode macroeconomic variables.
        
        Variables:
        - VIX (volatility index)
        - Credit spreads (investment grade, high yield)
        - Economic Policy Uncertainty (EPU)
        - Inflation expectations
        - Fed funds rate
        """
        
        def __init__(self, n_macro: int, d_model: int, dropout: float = 0.1):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(n_macro, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            
            # Time-varying macro attention
            self.macro_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            
        def forward(
            self, 
            macro: torch.Tensor, 
            price_embed: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Encode macro variables with optional price context.
            
            Args:
                macro: [batch, seq_len, n_macro]
                price_embed: Optional [batch, seq_len, d_model]
                
            Returns:
                macro_embed: [batch, seq_len, d_model]
            """
            macro_embed = self.encoder(macro)
            
            if price_embed is not None:
                # Cross-attention: macro attends to price dynamics
                macro_context, _ = self.macro_attention(
                    query=macro_embed,
                    key=price_embed,
                    value=price_embed
                )
                macro_embed = macro_embed + macro_context
                
            return macro_embed


    class TDAFusion(nn.Module):
        """
        Fuse TDA (Topological Data Analysis) features.
        
        TDA features include:
        - Persistence diagrams (H0, H1 homology)
        - Betti numbers
        - Entropy measures
        - Wasserstein distances
        """
        
        def __init__(self, n_tda: int, d_model: int, dropout: float = 0.1):
            super().__init__()
            
            # TDA encoder with residual
            self.tda_proj = nn.Linear(n_tda, d_model)
            
            self.tda_encoder = nn.Sequential(
                nn.Linear(n_tda, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            
            # Gated fusion with price features
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            
            self.norm = nn.LayerNorm(d_model)
            
        def forward(
            self, 
            tda: torch.Tensor, 
            price_embed: torch.Tensor
        ) -> torch.Tensor:
            """
            Fuse TDA features with price embeddings.
            
            Args:
                tda: [batch, seq_len, n_tda]
                price_embed: [batch, seq_len, d_model]
                
            Returns:
                fused: [batch, seq_len, d_model]
            """
            # Encode TDA
            tda_encoded = self.tda_encoder(tda)
            tda_residual = self.tda_proj(tda)
            tda_embed = tda_encoded + tda_residual
            
            # Gated fusion
            combined = torch.cat([price_embed, tda_embed], dim=-1)
            gate = self.gate(combined)
            
            fused = gate * price_embed + (1 - gate) * tda_embed
            
            return self.norm(fused)


    class TransformerEncoderBlock(nn.Module):
        """Single transformer encoder block with pre-norm."""
        
        def __init__(
            self,
            d_model: int,
            n_heads: int,
            ff_dim: int,
            dropout: float = 0.1,
            attention_dropout: float = 0.1
        ):
            super().__init__()
            
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, d_model),
                nn.Dropout(dropout),
            )
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(
            self, 
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward with attention weights output."""
            # Pre-norm self-attention
            x_norm = self.norm1(x)
            attn_out, attn_weights = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
                need_weights=True
            )
            x = x + self.dropout(attn_out)
            
            # Pre-norm FFN
            x = x + self.ffn(self.norm2(x))
            
            return x, attn_weights


    class QuantileHead(nn.Module):
        """Output head for quantile regression."""
        
        def __init__(self, d_model: int, n_quantiles: int = 5, forecast_horizon: int = 1):
            super().__init__()
            
            self.n_quantiles = n_quantiles
            self.forecast_horizon = forecast_horizon
            
            # Quantile values (e.g., 0.1, 0.25, 0.5, 0.75, 0.9)
            quantiles = torch.linspace(0.1, 0.9, n_quantiles)
            self.register_buffer('quantiles', quantiles)
            
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_quantiles * forecast_horizon),
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Predict quantiles.
            
            Args:
                x: [batch, d_model] - pooled representation
                
            Returns:
                quantiles: [batch, forecast_horizon, n_quantiles]
            """
            out = self.head(x)
            return out.view(-1, self.forecast_horizon, self.n_quantiles)
        
        def quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """
            Pinball loss for quantile regression.
            
            Args:
                pred: [batch, horizon, n_quantiles]
                target: [batch, horizon]
            """
            target = target.unsqueeze(-1)  # [batch, horizon, 1]
            errors = target - pred  # [batch, horizon, n_quantiles]
            
            loss = torch.max(
                self.quantiles * errors,
                (self.quantiles - 1) * errors
            )
            
            return loss.mean()


    class TemporalTransformer(nn.Module):
        """
        Complete Temporal Transformer for time series prediction.
        
        Combines:
        - Learnable positional encoding with seasonality
        - TDA feature fusion
        - Macroeconomic integration
        - Multi-head self-attention encoder
        - Quantile regression output
        """
        
        def __init__(self, config: TemporalTransformerConfig):
            super().__init__()
            
            self.config = config
            
            # Input projections
            self.price_proj = nn.Linear(config.price_features, config.model_dim)
            
            # Positional encoding
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=config.model_dim,
                max_len=config.max_seq_len,
                use_seasonal=config.use_seasonal_pos,
                dropout=config.dropout
            )
            
            # TDA fusion
            self.tda_fusion = TDAFusion(
                n_tda=config.tda_features,
                d_model=config.model_dim,
                dropout=config.dropout
            )
            
            # Macro encoder
            self.macro_encoder = MacroEncoder(
                n_macro=config.macro_features,
                d_model=config.model_dim,
                dropout=config.dropout
            )
            
            # Feature fusion
            self.feature_fusion = nn.Sequential(
                nn.Linear(config.model_dim * 3, config.model_dim),
                nn.LayerNorm(config.model_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            
            # Transformer encoder stack
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderBlock(
                    d_model=config.model_dim,
                    n_heads=config.num_heads,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout
                )
                for _ in range(config.num_encoder_layers)
            ])
            
            # Output layers
            self.output_norm = nn.LayerNorm(config.model_dim)
            
            # Multiple output heads
            self.point_head = nn.Linear(config.model_dim, config.output_dim * config.forecast_horizon)
            self.quantile_head = QuantileHead(
                config.model_dim, 
                config.num_quantiles, 
                config.forecast_horizon
            )
            self.volatility_head = nn.Linear(config.model_dim, config.forecast_horizon)
            
            self._init_weights()
            
        def _init_weights(self):
            """Initialize weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)
                    
        def forward(
            self,
            price: torch.Tensor,
            tda: Optional[torch.Tensor] = None,
            macro: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            return_attention: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                price: [batch, seq_len, price_features]
                tda: Optional [batch, seq_len, tda_features]
                macro: Optional [batch, seq_len, macro_features]
                mask: Optional attention mask
                return_attention: Whether to return attention weights
                
            Returns:
                Dictionary with predictions and optional attention
            """
            batch_size, seq_len, _ = price.shape
            
            # Encode price features
            price_embed = self.price_proj(price)
            price_embed = self.pos_encoding(price_embed)
            
            # Fuse TDA features
            if tda is not None:
                price_embed = self.tda_fusion(tda, price_embed)
            
            # Encode macro variables
            if macro is not None:
                macro_embed = self.macro_encoder(macro, price_embed)
            else:
                macro_embed = torch.zeros_like(price_embed)
            
            # Combine all features
            combined = torch.cat([
                price_embed,
                price_embed,  # Placeholder if TDA already fused
                macro_embed
            ], dim=-1)
            x = self.feature_fusion(combined)
            
            # Transformer encoder
            attention_weights = []
            for layer in self.encoder_layers:
                x, attn = layer(x, mask=mask)
                if return_attention:
                    attention_weights.append(attn)
            
            # Output normalization
            x = self.output_norm(x)
            
            # Pool to single representation (use last position or mean)
            pooled = x[:, -1, :]  # Last position
            
            # Generate predictions
            point_pred = self.point_head(pooled)
            point_pred = point_pred.view(batch_size, self.config.forecast_horizon, -1)
            
            quantile_pred = self.quantile_head(pooled)
            
            volatility_pred = F.softplus(self.volatility_head(pooled))
            
            result = {
                'point': point_pred.squeeze(-1),  # [batch, horizon]
                'quantiles': quantile_pred,       # [batch, horizon, n_quantiles]
                'volatility': volatility_pred,    # [batch, horizon]
                'embedding': pooled,              # [batch, d_model]
            }
            
            if return_attention:
                result['attention'] = attention_weights
                
            return result
        
        def compute_loss(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute combined loss.
            
            Args:
                predictions: Output from forward()
                targets: [batch, horizon] - actual returns
                
            Returns:
                loss: Combined loss
                metrics: Dictionary of individual losses
            """
            # Point prediction loss (MSE)
            point_loss = F.mse_loss(predictions['point'], targets)
            
            # Quantile loss
            quantile_loss = self.quantile_head.quantile_loss(
                predictions['quantiles'], targets
            )
            
            # Volatility-scaled loss (heteroscedastic)
            vol = predictions['volatility'] + 1e-6
            nll_loss = 0.5 * (torch.log(vol) + (targets - predictions['point'])**2 / vol)
            nll_loss = nll_loss.mean()
            
            # Combined loss
            loss = point_loss + 0.5 * quantile_loss + 0.1 * nll_loss
            
            metrics = {
                'point_loss': point_loss.item(),
                'quantile_loss': quantile_loss.item(),
                'nll_loss': nll_loss.item(),
                'total_loss': loss.item(),
                'mae': F.l1_loss(predictions['point'], targets).item(),
            }
            
            return loss, metrics
        
        def predict_with_uncertainty(
            self,
            price: torch.Tensor,
            tda: Optional[torch.Tensor] = None,
            macro: Optional[torch.Tensor] = None,
            n_samples: int = 20
        ) -> Dict[str, torch.Tensor]:
            """
            Monte Carlo dropout for uncertainty estimation.
            
            Args:
                price, tda, macro: Input tensors
                n_samples: Number of MC samples
                
            Returns:
                Dictionary with mean, std, and samples
            """
            self.train()  # Enable dropout
            
            samples = []
            for _ in range(n_samples):
                with torch.no_grad():
                    pred = self.forward(price, tda, macro)
                    samples.append(pred['point'])
            
            samples = torch.stack(samples, dim=0)  # [n_samples, batch, horizon]
            
            self.eval()
            
            return {
                'mean': samples.mean(dim=0),
                'std': samples.std(dim=0),
                'samples': samples,
                'lower_95': torch.quantile(samples, 0.025, dim=0),
                'upper_95': torch.quantile(samples, 0.975, dim=0),
            }


    class TemporalTransformerTrainer:
        """Training wrapper for Temporal Transformer."""
        
        def __init__(self, config: TemporalTransformerConfig, device: str = 'cpu'):
            self.config = config
            self.device = torch.device(device)
            
            self.model = TemporalTransformer(config).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=1e-5
            )
            
            # Cosine annealing with warmup
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_epochs,
                eta_min=config.learning_rate / 10
            )
            
            self.best_loss = float('inf')
            self.training_history = []
            self.best_state = None
            
        def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0.0
            metrics_sum = {}
            n_batches = 0
            
            for batch in dataloader:
                price = batch['price'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                tda = batch.get('tda')
                if tda is not None:
                    tda = tda.to(self.device)
                    
                macro = batch.get('macro')
                if macro is not None:
                    macro = macro.to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(price, tda, macro)
                loss, batch_metrics = self.model.compute_loss(predictions, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                n_batches += 1
            
            self.scheduler.step()
            
            metrics = {k: v / n_batches for k, v in metrics_sum.items()}
            metrics['lr'] = self.scheduler.get_last_lr()[0]
            
            return metrics
        
        def validate(self, dataloader: DataLoader) -> Dict[str, float]:
            """Validate on held-out data."""
            self.model.eval()
            total_loss = 0.0
            metrics_sum = {}
            n_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    price = batch['price'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    tda = batch.get('tda')
                    if tda is not None:
                        tda = tda.to(self.device)
                        
                    macro = batch.get('macro')
                    if macro is not None:
                        macro = macro.to(self.device)
                    
                    predictions = self.model(price, tda, macro)
                    loss, batch_metrics = self.model.compute_loss(predictions, targets)
                    
                    total_loss += loss.item()
                    for k, v in batch_metrics.items():
                        metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    n_batches += 1
            
            metrics = {k: v / n_batches for k, v in metrics_sum.items()}
            return metrics
        
        def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            patience: int = 20,
            verbose: bool = True
        ) -> Dict[str, Any]:
            """Full training loop."""
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                train_metrics = self.train_epoch(train_loader)
                
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    current_loss = val_metrics.get('total_loss', float('inf'))
                else:
                    val_metrics = {}
                    current_loss = train_metrics.get('total_loss', float('inf'))
                
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if verbose and epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train MAE={train_metrics.get('mae', 0):.5f}, "
                        f"Val MAE={val_metrics.get('mae', 0):.5f}, "
                        f"Best Loss={self.best_loss:.5f}"
                    )
                
                self.training_history.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics,
                })
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
            
            return {
                'best_loss': self.best_loss,
                'epochs_trained': epoch + 1,
                'history': self.training_history,
            }
        
        def predict(
            self,
            price: np.ndarray,
            tda: Optional[np.ndarray] = None,
            macro: Optional[np.ndarray] = None,
            with_uncertainty: bool = False
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Generate predictions."""
            self.model.eval()
            
            with torch.no_grad():
                price_tensor = torch.tensor(price, dtype=torch.float32).to(self.device)
                
                tda_tensor = None
                if tda is not None:
                    tda_tensor = torch.tensor(tda, dtype=torch.float32).to(self.device)
                    
                macro_tensor = None
                if macro is not None:
                    macro_tensor = torch.tensor(macro, dtype=torch.float32).to(self.device)
                
                if with_uncertainty:
                    result = self.model.predict_with_uncertainty(
                        price_tensor, tda_tensor, macro_tensor,
                        n_samples=self.config.mc_dropout_samples
                    )
                    pred = result['mean'].cpu().numpy()
                    info = {
                        'std': result['std'].cpu().numpy(),
                        'lower_95': result['lower_95'].cpu().numpy(),
                        'upper_95': result['upper_95'].cpu().numpy(),
                    }
                else:
                    result = self.model(price_tensor, tda_tensor, macro_tensor)
                    pred = result['point'].cpu().numpy()
                    info = {
                        'quantiles': result['quantiles'].cpu().numpy(),
                        'volatility': result['volatility'].cpu().numpy(),
                    }
            
            return pred, info
        
        def save(self, path: str):
            """Save model checkpoint."""
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config.to_dict(),
                'best_loss': self.best_loss,
                'training_history': self.training_history,
            }, path)
        
        def load(self, path: str):
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', [])

# Use appropriate implementation
if TORCH_AVAILABLE:
    pass  # PyTorch classes defined above
else:
    TemporalTransformer = FallbackTemporalTransformer
    TemporalTransformerTrainer = type('TemporalTransformerTrainer', (), {
        '__init__': lambda self, config, device='cpu': setattr(self, 'model', FallbackTemporalTransformer(config)),
        'predict': lambda self, *args, **kwargs: self.model.predict(*args, **kwargs),
        'fit': lambda self, *args, **kwargs: {'best_loss': 0, 'epochs_trained': 0},
    })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_synthetic_timeseries(
    n_samples: int = 500,
    seq_len: int = 60,
    n_price_features: int = 5,
    n_tda_features: int = 20,
    n_macro_features: int = 6,
    forecast_horizon: int = 5
) -> Dict[str, np.ndarray]:
    """Create synthetic time series data for testing."""
    np.random.seed(42)
    
    # Price features (OHLCV-like)
    price = np.random.randn(n_samples, seq_len, n_price_features) * 0.02
    
    # TDA features
    tda = np.random.randn(n_samples, seq_len, n_tda_features) * 0.1
    
    # Macro features
    macro = np.random.randn(n_samples, seq_len, n_macro_features) * 0.5
    
    # Targets (future returns)
    targets = np.random.randn(n_samples, forecast_horizon) * 0.02
    
    return {
        'price': price.astype(np.float32),
        'tda': tda.astype(np.float32),
        'macro': macro.astype(np.float32),
        'targets': targets.astype(np.float32),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = TemporalTransformerConfig(
        price_features=5,
        tda_features=20,
        macro_features=6,
        model_dim=64,
        num_heads=4,
        num_encoder_layers=2,
        max_epochs=5,
        forecast_horizon=5,
    )
    
    data = create_synthetic_timeseries(
        n_samples=200,
        seq_len=60,
        n_price_features=5,
        n_tda_features=20,
        n_macro_features=6,
        forecast_horizon=5
    )
    
    if TORCH_AVAILABLE:
        dataset = TensorDataset(
            torch.tensor(data['price']),
            torch.tensor(data['tda']),
            torch.tensor(data['macro']),
            torch.tensor(data['targets'])
        )
        
        def collate_fn(batch):
            return {
                'price': torch.stack([x[0] for x in batch]),
                'tda': torch.stack([x[1] for x in batch]),
                'macro': torch.stack([x[2] for x in batch]),
                'targets': torch.stack([x[3] for x in batch]),
            }
        
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        
        trainer = TemporalTransformerTrainer(config)
        result = trainer.fit(loader, verbose=True)
        
        print(f"\n✅ Training complete!")
        print(f"Best Loss: {result['best_loss']:.5f}")
        print(f"Epochs: {result['epochs_trained']}")
        
        # Test prediction
        pred, info = trainer.predict(data['price'][:1], data['tda'][:1], data['macro'][:1])
        print(f"Prediction shape: {pred.shape}")
    else:
        trainer = TemporalTransformerTrainer(config)
        pred, info = trainer.predict(data['price'][:1])
        print(f"\n✅ Fallback model working!")
        print(f"Prediction shape: {pred.shape}")
