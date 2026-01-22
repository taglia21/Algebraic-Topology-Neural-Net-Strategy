"""
Attention Factor Model for Statistical Arbitrage
=================================================

V2.3 Implementation of attention-based factor model that jointly learns
tradable arbitrage factors and optimal portfolio allocations.

Architecture:
- Multi-head cross-attention between assets and latent factors
- Conditional factor generation from firm characteristics
- Sharpe-aware loss function with transaction cost modeling
- End-to-end differentiable portfolio optimization

Key Features:
- Learns K tradable arbitrage factors from N assets
- Factors are interpretable via attention weights
- Optimizes net Sharpe ratio after realistic costs
- Supports both cross-sectional and time-series factors

Research Basis:
- Attention Factor Models achieve Sharpe 2.3-4+ (2024-2025 research)
- Joint factor/portfolio learning outperforms sequential approaches
- Transaction cost modeling critical for live trading

Target Performance:
- Sharpe ratio > 2.3 with 10bp/side transaction costs
- Max drawdown < 15% annualized
- Factor turnover < 50% monthly
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
from collections import deque

logger = logging.getLogger(__name__)

# Type checking imports (for static analysis only)
if TYPE_CHECKING:
    import torch as torch_typing
    Tensor = torch_typing.Tensor
else:
    # Define Tensor as Any for runtime when torch may not be available
    Tensor = Any

# Try to import PyTorch at runtime
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
    # Use actual torch.Tensor at runtime
    Tensor = torch.Tensor  # type: ignore[misc]
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class _DummyModule:
        pass
    nn = type('nn', (), {'Module': _DummyModule})()
    torch = None  # type: ignore[assignment]
    logger.warning("PyTorch not available - using simplified factor model fallback")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AttentionFactorConfig:
    """Configuration for Attention Factor Model."""
    
    # Asset universe
    n_assets: int = 100           # Number of assets in universe
    n_characteristics: int = 20   # Firm characteristics (size, momentum, etc.)
    
    # Factor structure
    n_factors: int = 8            # Number of latent factors
    factor_dim: int = 32          # Factor embedding dimension
    
    # Model architecture
    model_dim: int = 64           # Internal model dimension
    num_heads: int = 4            # Attention heads
    num_layers: int = 3           # Transformer encoder layers
    ff_dim: int = 128             # Feedforward dimension
    
    # Sequence parameters
    lookback: int = 60            # Days of price history
    
    # Regularization
    dropout: float = 0.1
    factor_orthogonality: float = 0.01  # Encourage uncorrelated factors
    weight_decay: float = 1e-5
    
    # Transaction costs
    transaction_cost_bps: float = 10.0   # 10bp per side
    turnover_penalty: float = 0.005      # Penalty for high turnover
    
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 200
    early_stopping_patience: int = 20
    
    # Risk constraints
    max_position_pct: float = 0.05       # 5% max per asset
    target_volatility: float = 0.15      # 15% annualized target vol
    leverage_limit: float = 2.0          # Max 2x leverage
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# FALLBACK IMPLEMENTATIONS (for non-PyTorch environments)
# =============================================================================

class FallbackAttentionFactorModel:
    """Fallback Attention Factor Model without PyTorch."""
    
    def __init__(self, config: AttentionFactorConfig):
        self.config = config
        logger.warning("Using simplified factor model (PyTorch not available)")
        
    def predict(
        self,
        price_history: np.ndarray,
        characteristics: np.ndarray,
        prev_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simple momentum-based factor model."""
        batch_size = price_history.shape[0]
        n_assets = price_history.shape[1]
        
        # Simple momentum signal from price history
        if price_history.ndim == 4:
            # [batch, n_assets, seq_len, features]
            prices = price_history[:, :, :, 0]  # Use first feature (close)
            returns = (prices[:, :, -1] - prices[:, :, 0]) / (prices[:, :, 0] + 1e-8)
        else:
            returns = np.random.randn(batch_size, n_assets) * 0.01
        
        # Normalize to weights
        weights = returns / (np.abs(returns).sum(axis=-1, keepdims=True) + 1e-8)
        weights = np.clip(weights, -self.config.max_position_pct, self.config.max_position_pct)
        
        # Mock factor loadings
        loadings = np.random.randn(batch_size, self.config.n_factors, n_assets) * 0.1
        
        return weights, {
            'method': 'fallback_momentum',
            'loadings': loadings,
            'factors': np.random.randn(batch_size, self.config.n_factors, self.config.model_dim)
        }


class FallbackAttentionFactorTrainer:
    """Fallback trainer for non-PyTorch environments."""
    
    def __init__(self, config: AttentionFactorConfig, device: str = 'cpu'):
        self.config = config
        self.model = FallbackAttentionFactorModel(config)
        self.best_sharpe = 0.0
        
    def fit(self, train_loader=None, val_loader=None, verbose=True):
        """Training not available without PyTorch."""
        logger.warning("Training not available without PyTorch")
        return {'best_sharpe': 0.0, 'epochs_trained': 0, 'history': []}
        
    def predict(
        self,
        price_history: np.ndarray,
        characteristics: np.ndarray,
        prev_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate portfolio weights."""
        return self.model.predict(price_history, characteristics, prev_weights)
    
    def save(self, path: str):
        """Save config to file."""
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f)
            
    def load(self, path: str):
        """Load config from file."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.config = AttentionFactorConfig(**data)


# =============================================================================
# PYTORCH IMPLEMENTATIONS
# =============================================================================

if TORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for temporal sequences."""
        
        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            self.register_buffer('pe', pe)
            
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Add positional encoding to input."""
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class CharacteristicEncoder(nn.Module):
        """Encode firm characteristics into dense embeddings."""
        
        def __init__(self, n_characteristics: int, embed_dim: int, dropout: float = 0.1):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(n_characteristics, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            
        def forward(self, characteristics: "torch.Tensor") -> "torch.Tensor":
            """Encode firm characteristics: [batch, n_assets, n_char] -> [batch, n_assets, embed_dim]"""
            return self.encoder(characteristics)


    class FactorAttentionLayer(nn.Module):
        """Cross-attention between assets and latent factors."""
        
        def __init__(
            self, 
            n_factors: int,
            model_dim: int, 
            num_heads: int = 4, 
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.n_factors = n_factors
            self.model_dim = model_dim
            
            # Learnable factor queries
            self.factor_queries = nn.Parameter(
                torch.randn(1, n_factors, model_dim) * 0.02
            )
            
            # Cross-attention: factors attend to assets
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Self-attention among factors
            self.self_attention = nn.MultiheadAttention(
                embed_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Feedforward
            self.ffn = nn.Sequential(
                nn.Linear(model_dim, model_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim * 4, model_dim),
                nn.Dropout(dropout),
            )
            
            self.norm1 = nn.LayerNorm(model_dim)
            self.norm2 = nn.LayerNorm(model_dim)
            self.norm3 = nn.LayerNorm(model_dim)
            
        def forward(
            self, 
            asset_embeddings: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute factor representations and loadings."""
            batch_size = asset_embeddings.size(0)
            
            # Expand factor queries for batch
            queries = self.factor_queries.expand(batch_size, -1, -1)
            
            # Cross-attention: factors query assets
            factors, attention_weights = self.cross_attention(
                query=queries,
                key=asset_embeddings,
                value=asset_embeddings,
                key_padding_mask=mask,
                need_weights=True
            )
            factors = self.norm1(queries + factors)
            
            # Self-attention among factors
            factors_sa, _ = self.self_attention(
                query=factors, key=factors, value=factors, need_weights=False
            )
            factors = self.norm2(factors + factors_sa)
            
            # Feedforward
            factors = self.norm3(factors + self.ffn(factors))
            
            return factors, attention_weights


    class TemporalEncoder(nn.Module):
        """Encode price/volume time series dynamics per asset."""
        
        def __init__(
            self,
            input_dim: int,
            model_dim: int,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.input_proj = nn.Linear(input_dim, model_dim)
            self.pos_encoding = PositionalEncoding(model_dim, dropout=dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.output_proj = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.LayerNorm(model_dim),
            )
            
        def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            """Encode temporal dynamics: [batch, n_assets, seq_len, features] -> [batch, n_assets, model_dim]"""
            batch_size, n_assets, seq_len, input_dim = x.shape
            
            x = x.view(batch_size * n_assets, seq_len, input_dim)
            x = self.input_proj(x)
            x = self.pos_encoding(x)
            x = self.encoder(x, src_key_padding_mask=mask)
            x = x[:, -1, :]  # Use last timestep
            x = x.view(batch_size, n_assets, -1)
            
            return self.output_proj(x)


    class PortfolioOptimizer(nn.Module):
        """Differentiable portfolio optimization layer."""
        
        def __init__(
            self,
            n_factors: int,
            n_assets: int,
            model_dim: int,
            max_position: float = 0.05,
            leverage_limit: float = 2.0
        ):
            super().__init__()
            
            self.n_factors = n_factors
            self.n_assets = n_assets
            self.max_position = max_position
            self.leverage_limit = leverage_limit
            
            self.factor_return_head = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.GELU(),
                nn.Linear(model_dim // 2, 1),
            )
            
            self.factor_cov_head = nn.Sequential(
                nn.Linear(model_dim * n_factors, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, n_factors * (n_factors + 1) // 2),
            )
            
            self.weight_refiner = nn.Sequential(
                nn.Linear(n_assets, n_assets * 2),
                nn.GELU(),
                nn.Linear(n_assets * 2, n_assets),
            )
            
        def forward(
            self,
            factors: "torch.Tensor",
            loadings: "torch.Tensor",
            prev_weights: Optional["torch.Tensor"] = None
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            """Compute optimal portfolio weights."""
            batch_size = factors.size(0)
            
            factor_returns = self.factor_return_head(factors).squeeze(-1)
            
            factor_flat = factors.view(batch_size, -1)
            cov_params = self.factor_cov_head(factor_flat)
            
            L = torch.zeros(batch_size, self.n_factors, self.n_factors, device=factors.device)
            idx = torch.tril_indices(self.n_factors, self.n_factors)
            L[:, idx[0], idx[1]] = cov_params
            L = L + 0.01 * torch.eye(self.n_factors, device=factors.device)
            factor_cov = torch.bmm(L, L.transpose(1, 2))
            
            asset_returns = torch.bmm(loadings.transpose(1, 2), factor_returns.unsqueeze(-1)).squeeze(-1)
            
            raw_weights = self.weight_refiner(asset_returns)
            weights = self._apply_constraints(raw_weights, prev_weights)
            
            info = {
                'factor_returns': factor_returns,
                'factor_cov': factor_cov,
                'asset_returns': asset_returns,
                'raw_weights': raw_weights,
            }
            
            return weights, info
        
        def _apply_constraints(
            self,
            weights: "torch.Tensor",
            prev_weights: Optional["torch.Tensor"] = None
        ) -> "torch.Tensor":
            """Apply position and leverage constraints."""
            weights = torch.clamp(weights, -self.max_position, self.max_position)
            total_leverage = torch.abs(weights).sum(dim=-1, keepdim=True)
            scale = torch.clamp(self.leverage_limit / (total_leverage + 1e-8), max=1.0)
            weights = weights * scale
            return weights


    class AttentionFactorModel(nn.Module):
        """Complete Attention Factor Model for Statistical Arbitrage."""
        
        def __init__(self, config: AttentionFactorConfig):
            super().__init__()
            
            self.config = config
            
            self.characteristic_encoder = CharacteristicEncoder(
                n_characteristics=config.n_characteristics,
                embed_dim=config.model_dim,
                dropout=config.dropout
            )
            
            self.temporal_encoder = TemporalEncoder(
                input_dim=5,
                model_dim=config.model_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            
            self.asset_fusion = nn.Sequential(
                nn.Linear(config.model_dim * 2, config.model_dim),
                nn.LayerNorm(config.model_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            
            self.factor_layers = nn.ModuleList([
                FactorAttentionLayer(
                    n_factors=config.n_factors,
                    model_dim=config.model_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout
                )
                for _ in range(2)
            ])
            
            self.portfolio_optimizer = PortfolioOptimizer(
                n_factors=config.n_factors,
                n_assets=config.n_assets,
                model_dim=config.model_dim,
                max_position=config.max_position_pct,
                leverage_limit=config.leverage_limit
            )
            
            self._init_weights()
            
        def _init_weights(self):
            """Initialize model weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                    
        def forward(
            self,
            price_history: "torch.Tensor",
            characteristics: "torch.Tensor",
            prev_weights: Optional["torch.Tensor"] = None,
            mask: Optional["torch.Tensor"] = None
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            """Forward pass."""
            temporal_embed = self.temporal_encoder(price_history)
            char_embed = self.characteristic_encoder(characteristics)
            
            asset_embed = self.asset_fusion(
                torch.cat([temporal_embed, char_embed], dim=-1)
            )
            
            factors = None
            loadings = None
            attention_weights_list = []
            
            for layer in self.factor_layers:
                factors, loadings = layer(asset_embed, mask)
                attention_weights_list.append(loadings)
                
            weights, opt_info = self.portfolio_optimizer(factors, loadings, prev_weights)
            
            info = {
                'factors': factors,
                'loadings': loadings,
                'attention_weights': attention_weights_list,
                'temporal_embed': temporal_embed,
                'char_embed': char_embed,
                **opt_info
            }
            
            return weights, info
        
        def compute_sharpe_loss(
            self,
            weights: "torch.Tensor",
            returns: "torch.Tensor",
            prev_weights: Optional["torch.Tensor"] = None,
            info: Optional[Dict] = None
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Compute Sharpe-aware loss with transaction costs."""
            portfolio_return = (weights * returns).sum(dim=-1)
            
            if prev_weights is not None:
                turnover = torch.abs(weights - prev_weights).sum(dim=-1)
                tc = turnover * self.config.transaction_cost_bps / 10000 * 2
            else:
                turnover = torch.abs(weights).sum(dim=-1)
                tc = turnover * self.config.transaction_cost_bps / 10000
                
            net_return = portfolio_return - tc
            
            mean_return = net_return.mean()
            std_return = net_return.std() + 1e-8
            sharpe = mean_return / std_return * np.sqrt(252)
            
            orthogonality_loss = torch.tensor(0.0, device=weights.device)
            if info is not None and 'loadings' in info:
                loadings = info['loadings']
                gram = torch.bmm(loadings, loadings.transpose(1, 2))
                n_factors = loadings.size(1)
                identity = torch.eye(n_factors, device=loadings.device).unsqueeze(0)
                orthogonality_loss = ((gram - identity) ** 2).mean()
                
            turnover_loss = turnover.mean() * self.config.turnover_penalty
            
            loss = -sharpe + self.config.factor_orthogonality * orthogonality_loss + turnover_loss
            
            metrics = {
                'sharpe': sharpe.item(),
                'mean_return': mean_return.item(),
                'std_return': std_return.item(),
                'transaction_cost': tc.mean().item(),
                'turnover': turnover.mean().item(),
                'orthogonality_loss': orthogonality_loss.item(),
            }
            
            return loss, metrics


    class AttentionFactorTrainer:
        """Training wrapper for Attention Factor Model."""
        
        def __init__(self, config: AttentionFactorConfig, device: str = 'cpu'):
            self.config = config
            self.device = torch.device(device)
            
            self.model = AttentionFactorModel(config).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_epochs,
                eta_min=config.learning_rate / 10
            )
            
            self.best_sharpe = -float('inf')
            self.patience_counter = 0
            self.training_history = []
            self.best_state = None
            
        def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0.0
            metrics_sum = {}
            n_batches = 0
            
            for batch in dataloader:
                price_history = batch['price_history'].to(self.device)
                characteristics = batch['characteristics'].to(self.device)
                returns = batch['returns'].to(self.device)
                batch_size = price_history.size(0)
                
                self.optimizer.zero_grad()
                
                # No prev_weights in training (each batch is independent)
                weights, info = self.model(price_history, characteristics, prev_weights=None)
                loss, batch_metrics = self.model.compute_sharpe_loss(weights, returns, None, info)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                n_batches += 1
                
            self.scheduler.step()
            
            metrics = {k: v / n_batches for k, v in metrics_sum.items()}
            metrics['loss'] = total_loss / n_batches
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
                    price_history = batch['price_history'].to(self.device)
                    characteristics = batch['characteristics'].to(self.device)
                    returns = batch['returns'].to(self.device)
                    
                    weights, info = self.model(price_history, characteristics, prev_weights=None)
                    loss, batch_metrics = self.model.compute_sharpe_loss(weights, returns, None, info)
                    
                    total_loss += loss.item()
                    for k, v in batch_metrics.items():
                        metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    n_batches += 1
                    
            metrics = {k: v / n_batches for k, v in metrics_sum.items()}
            metrics['loss'] = total_loss / n_batches
            
            return metrics
        
        def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            verbose: bool = True
        ) -> Dict[str, Any]:
            """Full training loop with early stopping."""
            
            for epoch in range(self.config.max_epochs):
                train_metrics = self.train_epoch(train_loader, epoch)
                
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    current_sharpe = val_metrics.get('sharpe', -float('inf'))
                else:
                    current_sharpe = train_metrics.get('sharpe', -float('inf'))
                    
                if current_sharpe > self.best_sharpe:
                    self.best_sharpe = current_sharpe
                    self.patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    self.patience_counter += 1
                    
                if verbose and epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train Sharpe={train_metrics.get('sharpe', 0):.3f}, "
                        f"Val Sharpe={val_metrics.get('sharpe', 0):.3f}, "
                        f"Best={self.best_sharpe:.3f}"
                    )
                    
                self.training_history.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics,
                })
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
                
            return {
                'best_sharpe': self.best_sharpe,
                'epochs_trained': epoch + 1,
                'history': self.training_history,
            }
        
        def predict(
            self,
            price_history: np.ndarray,
            characteristics: np.ndarray,
            prev_weights: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Generate portfolio weights from input data."""
            self.model.eval()
            
            with torch.no_grad():
                price_tensor = torch.tensor(price_history, dtype=torch.float32).to(self.device)
                char_tensor = torch.tensor(characteristics, dtype=torch.float32).to(self.device)
                
                if prev_weights is not None:
                    prev_tensor = torch.tensor(prev_weights, dtype=torch.float32).to(self.device)
                else:
                    prev_tensor = None
                    
                weights, info = self.model(price_tensor, char_tensor, prev_tensor)
                
                numpy_info = {}
                for k, v in info.items():
                    if isinstance(v, torch.Tensor):
                        numpy_info[k] = v.cpu().numpy()
                    elif isinstance(v, list):
                        numpy_info[k] = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in v]
                    else:
                        numpy_info[k] = v
                        
                return weights.cpu().numpy(), numpy_info
        
        def save(self, path: str):
            """Save model checkpoint."""
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config.to_dict(),
                'best_sharpe': self.best_sharpe,
                'training_history': self.training_history,
            }, path)
            
        def load(self, path: str):
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.best_sharpe = checkpoint.get('best_sharpe', -float('inf'))
            self.training_history = checkpoint.get('training_history', [])

# Use appropriate implementation
if TORCH_AVAILABLE:
    # PyTorch classes are defined above
    pass
else:
    # Use fallback implementations
    AttentionFactorModel = FallbackAttentionFactorModel
    AttentionFactorTrainer = FallbackAttentionFactorTrainer


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_synthetic_data(
    n_samples: int = 1000,
    n_assets: int = 100,
    seq_len: int = 60,
    n_features: int = 5,
    n_characteristics: int = 20
) -> Dict[str, np.ndarray]:
    """Create synthetic data for testing."""
    np.random.seed(42)
    
    price_history = np.random.randn(n_samples, n_assets, seq_len, n_features) * 0.02
    characteristics = np.random.randn(n_samples, n_assets, n_characteristics)
    returns = np.random.randn(n_samples, n_assets) * 0.02
    
    return {
        'price_history': price_history.astype(np.float32),
        'characteristics': characteristics.astype(np.float32),
        'returns': returns.astype(np.float32),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = AttentionFactorConfig(
        n_assets=50,
        n_characteristics=20,
        n_factors=5,
        model_dim=32,
        num_layers=2,
        max_epochs=5,
    )
    
    data = create_synthetic_data(
        n_samples=200,
        n_assets=50,
        seq_len=60,
        n_features=5,
        n_characteristics=20
    )
    
    if TORCH_AVAILABLE:
        dataset = TensorDataset(
            torch.tensor(data['price_history']),
            torch.tensor(data['characteristics']),
            torch.tensor(data['returns'])
        )
        
        def collate_fn(batch):
            return {
                'price_history': torch.stack([x[0] for x in batch]),
                'characteristics': torch.stack([x[1] for x in batch]),
                'returns': torch.stack([x[2] for x in batch]),
            }
        
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        
        trainer = AttentionFactorTrainer(config)
        result = trainer.fit(loader, verbose=True)
        
        print(f"\n✅ Training complete!")
        print(f"Best Sharpe: {result['best_sharpe']:.3f}")
        print(f"Epochs: {result['epochs_trained']}")
    else:
        # Test fallback
        trainer = AttentionFactorTrainer(config)
        weights, info = trainer.predict(
            data['price_history'][:1],
            data['characteristics'][:1]
        )
        print(f"\n✅ Fallback model working!")
        print(f"Weights shape: {weights.shape}")
        print(f"Method: {info.get('method', 'unknown')}")
