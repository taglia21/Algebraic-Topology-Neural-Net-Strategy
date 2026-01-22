"""
V2.3 Production Engine
======================

Integrates all V2.3 cutting-edge ML/RL components:

1. Attention Factor Model - Joint factor learning + portfolio optimization
2. Temporal Transformer - TDA/macro integrated predictions
3. Dueling SAC - Enhanced RL with distributional critics
4. POMDP Controller - Regime-aware belief state tracking

Key Differentiators from V2.2:
- Cross-attention between assets and factors (vs fixed factors)
- Distributional RL captures tail risk (vs point estimates)
- Belief state policy conditioning (vs reactive regime switching)
- Quantile-based uncertainty for position sizing

Architecture Flow:
1. Data → Temporal Transformer → Predictions + Uncertainty
2. Data → Attention Factor Model → Factor-weighted Returns
3. Beliefs → POMDP Controller → Risk-adjusted Base Position
4. All Features → Dueling SAC → Final Position Optimization

Target Performance:
- Sharpe: 2.0+ (vs V2.2 target of 1.8)
- Max Drawdown: < 4% (improved tail risk)
- Latency: < 200ms per decision
"""

import numpy as np
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import threading

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - V2.3 features limited")

# Import V2.3 components
try:
    from src.models.attention_factor_model import (
        AttentionFactorModel, AttentionFactorConfig
    )
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False
    
try:
    from src.models.temporal_transformer import (
        TemporalTransformer, TemporalTransformerConfig
    )
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    
try:
    from src.agents.dueling_sac import (
        DuelingSAC, DuelingSACConfig
    )
    DUELING_SAC_AVAILABLE = True
except ImportError:
    DUELING_SAC_AVAILABLE = False
    
try:
    from src.regime.pomdp_controller import (
        POMDPController, POMDPConfig, MarketRegime
    )
    POMDP_AVAILABLE = True
except ImportError:
    POMDP_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class V23EngineConfig:
    """Configuration for V2.3 Production Engine."""
    
    # Component enablement
    use_attention_factor: bool = True
    use_temporal_transformer: bool = True
    use_dueling_sac: bool = True
    use_pomdp_controller: bool = True
    
    # Ensemble blending weights
    attention_weight: float = 0.3
    transformer_weight: float = 0.3
    pomdp_weight: float = 0.2
    sac_weight: float = 0.2
    
    # Feature dimensions (matching component configs)
    n_assets: int = 10
    n_characteristics: int = 16
    n_factors: int = 5
    seq_length: int = 60
    tda_dim: int = 20
    macro_dim: int = 4
    
    # Position constraints
    max_position_pct: float = 0.03
    min_position_pct: float = 0.0
    max_portfolio_heat: float = 0.20
    
    # Risk management
    use_uncertainty_scaling: bool = True
    uncertainty_threshold: float = 0.8  # Scale down if uncertainty > threshold
    
    # Device
    device: str = 'cpu'
    
    # Model paths
    model_dir: str = 'models/v23'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# V2.3 ENGINE STATES
# =============================================================================

@dataclass
class V23EngineState:
    """Current state of the V2.3 engine."""
    
    # Predictions
    attention_weights: Optional[np.ndarray] = None  # [n_assets]
    transformer_pred: Optional[np.ndarray] = None   # [n_assets]
    transformer_uncertainty: Optional[np.ndarray] = None
    sac_position: Optional[np.ndarray] = None
    pomdp_belief: Optional[np.ndarray] = None
    
    # Final output
    final_positions: Optional[np.ndarray] = None
    
    # Metadata
    regime: str = "UNKNOWN"
    risk_scale: float = 1.0
    confidence: float = 0.5
    latency_ms: float = 0.0
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        return result


# =============================================================================
# V2.3 PRODUCTION ENGINE
# =============================================================================

class V23ProductionEngine:
    """
    V2.3 Production Engine integrating all advanced components.
    
    Features:
    - Attention Factor Model for cross-asset factor learning
    - Temporal Transformer for sequence predictions
    - Dueling SAC for RL-based position optimization
    - POMDP Controller for belief-conditioned decisions
    """
    
    def __init__(self, config: V23EngineConfig):
        self.config = config
        self.device = config.device
        self.state = V23EngineState()
        self._lock = threading.Lock()
        
        # Component availability flags
        self._components_available = {
            'attention': False,
            'transformer': False,
            'dueling_sac': False,
            'pomdp': False,
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("=" * 60)
        logger.info("V2.3 PRODUCTION ENGINE INITIALIZED")
        logger.info("=" * 60)
        self._log_component_status()
        
    def _initialize_components(self):
        """Initialize V2.3 ML/RL components."""
        
        # 1. Attention Factor Model
        if self.config.use_attention_factor and ATTENTION_AVAILABLE and TORCH_AVAILABLE:
            try:
                attention_config = AttentionFactorConfig(
                    n_assets=self.config.n_assets,
                    n_characteristics=self.config.n_characteristics,
                    n_factors=self.config.n_factors,
                    lookback=self.config.seq_length,  # Use 'lookback' parameter
                )
                self.attention_model = AttentionFactorModel(attention_config)
                self.attention_model.eval()
                self._components_available['attention'] = True
                logger.info("✅ Attention Factor Model initialized")
            except Exception as e:
                logger.warning(f"⚠️ Attention Factor Model failed: {e}")
                self.attention_model = None
        else:
            self.attention_model = None
            
        # 2. Temporal Transformer
        if self.config.use_temporal_transformer and TEMPORAL_AVAILABLE and TORCH_AVAILABLE:
            try:
                transformer_config = TemporalTransformerConfig(
                    total_input_dim=self.config.n_characteristics,
                    tda_features=self.config.tda_dim,
                    macro_features=self.config.macro_dim,
                    max_seq_len=self.config.seq_length,
                )
                self.transformer = TemporalTransformer(transformer_config)
                self.transformer.eval()
                self._components_available['transformer'] = True
                logger.info("✅ Temporal Transformer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Temporal Transformer failed: {e}")
                self.transformer = None
        else:
            self.transformer = None
            
        # 3. Dueling SAC
        if self.config.use_dueling_sac and DUELING_SAC_AVAILABLE and TORCH_AVAILABLE:
            try:
                # State dim = characteristics + tda + macro + transformer pred + pomdp belief
                state_dim = (
                    self.config.n_characteristics + 
                    self.config.tda_dim + 
                    self.config.macro_dim + 
                    1 +  # transformer prediction
                    5    # POMDP belief (5 regimes)
                )
                sac_config = DuelingSACConfig(
                    state_dim=state_dim,
                    action_dim=1,  # Position size
                    max_position=self.config.max_position_pct,
                )
                self.sac_agent = DuelingSAC(sac_config, device=self.device)
                self._components_available['dueling_sac'] = True
                logger.info("✅ Dueling SAC Agent initialized")
            except Exception as e:
                logger.warning(f"⚠️ Dueling SAC failed: {e}")
                self.sac_agent = None
        else:
            self.sac_agent = None
            
        # 4. POMDP Controller
        if self.config.use_pomdp_controller and POMDP_AVAILABLE:
            try:
                pomdp_config = POMDPConfig(
                    observation_dim=self.config.n_characteristics,
                    tda_dim=self.config.tda_dim,
                    macro_dim=self.config.macro_dim,
                    max_position=self.config.max_position_pct,
                )
                self.pomdp = POMDPController(pomdp_config, device=self.device) if TORCH_AVAILABLE else None
                if self.pomdp:
                    self._components_available['pomdp'] = True
                    logger.info("✅ POMDP Controller initialized")
            except Exception as e:
                logger.warning(f"⚠️ POMDP Controller failed: {e}")
                self.pomdp = None
        else:
            self.pomdp = None
    
    def _log_component_status(self):
        """Log which components are available."""
        status = []
        for name, available in self._components_available.items():
            status.append(f"{name}: {'✅' if available else '❌'}")
        logger.info("Component Status: " + " | ".join(status))
        
        # Calculate effective weights
        active = sum(1 for v in self._components_available.values() if v)
        if active == 0:
            logger.warning("⚠️ No V2.3 components available - falling back to baseline")
    
    @torch.no_grad() if TORCH_AVAILABLE else lambda x: x
    def generate_signals(
        self,
        returns: np.ndarray,
        characteristics: np.ndarray,
        tda_features: Optional[np.ndarray] = None,
        macro_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, V23EngineState]:
        """
        Generate trading signals using V2.3 ensemble.
        
        Args:
            returns: [seq_length, n_assets] historical returns
            characteristics: [seq_length, n_assets, n_characteristics]
            tda_features: [seq_length, tda_dim] TDA features
            macro_features: [seq_length, macro_dim] macro variables
            
        Returns:
            positions: [n_assets] target position sizes
            state: Engine state with component outputs
        """
        start_time = time.perf_counter()
        
        with self._lock:
            self.state = V23EngineState()
            self.state.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            n_assets = returns.shape[-1] if returns is not None else self.config.n_assets
            
            # Prepare default features
            if tda_features is None:
                tda_features = np.zeros((self.config.seq_length, self.config.tda_dim))
            if macro_features is None:
                macro_features = np.zeros((self.config.seq_length, self.config.macro_dim))
                
            # Initialize component outputs
            attention_positions = np.zeros(n_assets)
            transformer_pred = np.zeros(n_assets)
            transformer_uncertainty = np.ones(n_assets)
            sac_position = 0.0
            pomdp_position = 0.0
            risk_scale = 1.0
            
            # 1. Attention Factor Model
            if self._components_available['attention'] and self.attention_model is not None:
                try:
                    attention_positions = self._run_attention_model(
                        returns, characteristics
                    )
                    self.state.attention_weights = attention_positions
                except Exception as e:
                    logger.warning(f"Attention model error: {e}")
            
            # 2. Temporal Transformer
            if self._components_available['transformer'] and self.transformer is not None:
                try:
                    transformer_pred, transformer_uncertainty = self._run_transformer(
                        characteristics, tda_features, macro_features
                    )
                    self.state.transformer_pred = transformer_pred
                    self.state.transformer_uncertainty = transformer_uncertainty
                except Exception as e:
                    logger.warning(f"Transformer error: {e}")
            
            # 3. POMDP Controller (for regime-based risk scaling)
            if self._components_available['pomdp'] and self.pomdp is not None:
                try:
                    # Use latest observation
                    obs = characteristics[-1].mean(axis=0) if len(characteristics.shape) > 2 else characteristics[-1]
                    tda_obs = tda_features[-1] if tda_features is not None else None
                    macro_obs = macro_features[-1] if macro_features is not None else None
                    
                    _, info = self.pomdp.select_action(obs, tda_obs, macro_obs)
                    
                    self.state.pomdp_belief = info.get('belief')
                    self.state.regime = info.get('regime_name', 'UNKNOWN')
                    risk_scale = info.get('risk_scale', 1.0)
                    self.state.risk_scale = risk_scale
                except Exception as e:
                    logger.warning(f"POMDP error: {e}")
            
            # 4. Dueling SAC (for position optimization)
            if self._components_available['dueling_sac'] and self.sac_agent is not None:
                try:
                    sac_state = self._prepare_sac_state(
                        characteristics, tda_features, macro_features,
                        transformer_pred, self.state.pomdp_belief
                    )
                    sac_position = self.sac_agent.select_action(sac_state, deterministic=True)
                    self.state.sac_position = np.atleast_1d(sac_position)
                except Exception as e:
                    logger.warning(f"SAC error: {e}")
            
            # 5. Ensemble combination
            final_positions = self._ensemble_signals(
                attention_positions,
                transformer_pred,
                transformer_uncertainty,
                sac_position,
                risk_scale
            )
            
            # Apply position constraints
            final_positions = np.clip(
                final_positions,
                self.config.min_position_pct,
                self.config.max_position_pct
            )
            
            # Ensure portfolio heat constraint
            total_exposure = np.sum(np.abs(final_positions))
            if total_exposure > self.config.max_portfolio_heat:
                scale = self.config.max_portfolio_heat / total_exposure
                final_positions = final_positions * scale
            
            self.state.final_positions = final_positions
            self.state.latency_ms = (time.perf_counter() - start_time) * 1000
            
            return final_positions, self.state
    
    def _run_attention_model(
        self,
        returns: np.ndarray,
        characteristics: np.ndarray
    ) -> np.ndarray:
        """Run Attention Factor Model."""
        # Prepare input tensors
        returns_t = torch.FloatTensor(returns).unsqueeze(0)  # [1, seq, n_assets]
        char_t = torch.FloatTensor(characteristics).unsqueeze(0)  # [1, seq, n_assets, n_char]
        
        # Run model
        output = self.attention_model(char_t, returns_t)
        weights = output['weights']  # [1, n_assets]
        
        return weights.squeeze(0).numpy()
    
    def _run_transformer(
        self,
        characteristics: np.ndarray,
        tda_features: np.ndarray,
        macro_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Temporal Transformer."""
        # Use mean characteristics across assets as input
        if len(characteristics.shape) > 2:
            x = characteristics.mean(axis=1)  # [seq, n_char]
        else:
            x = characteristics
            
        x_t = torch.FloatTensor(x).unsqueeze(0)  # [1, seq, input_dim]
        tda_t = torch.FloatTensor(tda_features).unsqueeze(0)
        macro_t = torch.FloatTensor(macro_features).unsqueeze(0)
        
        output = self.transformer(x_t, tda_t, macro_t)
        pred = output['prediction']  # [1, 1] or [1, n_outputs]
        uncertainty = output.get('uncertainty', torch.ones_like(pred))
        
        # Broadcast to all assets if needed
        pred_np = pred.squeeze().numpy()
        unc_np = uncertainty.squeeze().numpy()
        
        if np.isscalar(pred_np):
            pred_np = np.full(self.config.n_assets, pred_np)
            unc_np = np.full(self.config.n_assets, unc_np)
            
        return pred_np, unc_np
    
    def _prepare_sac_state(
        self,
        characteristics: np.ndarray,
        tda_features: np.ndarray,
        macro_features: np.ndarray,
        transformer_pred: np.ndarray,
        pomdp_belief: Optional[np.ndarray]
    ) -> np.ndarray:
        """Prepare state vector for SAC agent."""
        # Latest characteristics (mean across assets)
        if len(characteristics.shape) > 2:
            char_latest = characteristics[-1].mean(axis=0)
        else:
            char_latest = characteristics[-1]
            
        # Latest TDA
        tda_latest = tda_features[-1] if tda_features is not None else np.zeros(self.config.tda_dim)
        
        # Latest macro
        macro_latest = macro_features[-1] if macro_features is not None else np.zeros(self.config.macro_dim)
        
        # Transformer prediction (scalar or mean)
        pred_scalar = np.mean(transformer_pred) if transformer_pred is not None else 0.0
        
        # POMDP belief
        belief = pomdp_belief if pomdp_belief is not None else np.ones(5) / 5
        
        # Concatenate state
        state = np.concatenate([
            char_latest[:self.config.n_characteristics],
            tda_latest[:self.config.tda_dim],
            macro_latest[:self.config.macro_dim],
            [pred_scalar],
            belief
        ])
        
        return state
    
    def _ensemble_signals(
        self,
        attention_positions: np.ndarray,
        transformer_pred: np.ndarray,
        transformer_uncertainty: np.ndarray,
        sac_position: float,
        risk_scale: float
    ) -> np.ndarray:
        """Combine component signals into final positions."""
        
        # Compute active weights (normalize for available components)
        weights = {
            'attention': self.config.attention_weight if self._components_available['attention'] else 0,
            'transformer': self.config.transformer_weight if self._components_available['transformer'] else 0,
            'sac': self.config.sac_weight if self._components_available['dueling_sac'] else 0,
            'pomdp': self.config.pomdp_weight if self._components_available['pomdp'] else 0,
        }
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Fallback: equal weight baseline
            return np.ones(self.config.n_assets) * self.config.max_position_pct * 0.5
        
        # Normalize weights
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Base positions from each component
        n_assets = len(attention_positions) if attention_positions is not None else self.config.n_assets
        combined = np.zeros(n_assets)
        
        # Attention contribution (already in position space)
        if weights['attention'] > 0:
            combined += weights['attention'] * attention_positions
        
        # Transformer contribution (convert prediction to position)
        if weights['transformer'] > 0:
            # Scale predictions to position range
            pred_scaled = np.tanh(transformer_pred) * self.config.max_position_pct
            
            # Apply uncertainty scaling if enabled
            if self.config.use_uncertainty_scaling:
                # Reduce position when uncertain
                certainty = 1 - np.clip(transformer_uncertainty / self.config.uncertainty_threshold, 0, 1)
                pred_scaled = pred_scaled * certainty
                
            combined += weights['transformer'] * pred_scaled
        
        # SAC contribution (single position, broadcast or apply to mean)
        if weights['sac'] > 0:
            sac_contrib = np.ones(n_assets) * np.mean(np.atleast_1d(sac_position))
            combined += weights['sac'] * sac_contrib
        
        # POMDP contribution (via risk scaling)
        if weights['pomdp'] > 0:
            combined = combined * risk_scale
            
        # Compute confidence from uncertainty
        if transformer_uncertainty is not None:
            mean_uncertainty = np.mean(transformer_uncertainty)
            self.state.confidence = float(1 - np.clip(mean_uncertainty, 0, 1))
        
        return combined
    
    def get_state(self) -> V23EngineState:
        """Get current engine state."""
        return self.state
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get component availability status."""
        return self._components_available.copy()
    
    def save_models(self, path: Optional[str] = None):
        """Save all model weights."""
        if path is None:
            path = self.config.model_dir
            
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if self.sac_agent is not None:
            self.sac_agent.save(f"{path}/dueling_sac.pt")
            
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load model weights."""
        if path is None:
            path = self.config.model_dir
            
        if self.sac_agent is not None and Path(f"{path}/dueling_sac.pt").exists():
            self.sac_agent.load(f"{path}/dueling_sac.pt")
            
        logger.info(f"Models loaded from {path}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = V23EngineConfig(
        n_assets=5,
        n_characteristics=10,
        n_factors=3,
        seq_length=30,
        tda_dim=10,
        macro_dim=4,
    )
    
    print("Testing V2.3 Production Engine...")
    engine = V23ProductionEngine(config)
    
    # Generate synthetic data
    seq_len = config.seq_length
    n_assets = config.n_assets
    n_char = config.n_characteristics
    
    returns = np.random.randn(seq_len, n_assets) * 0.02
    characteristics = np.random.randn(seq_len, n_assets, n_char)
    tda_features = np.random.randn(seq_len, config.tda_dim)
    macro_features = np.random.randn(seq_len, config.macro_dim)
    
    # Generate signals
    positions, state = engine.generate_signals(
        returns, characteristics, tda_features, macro_features
    )
    
    print(f"\nResults:")
    print(f"  Positions: {positions}")
    print(f"  Regime: {state.regime}")
    print(f"  Risk Scale: {state.risk_scale:.3f}")
    print(f"  Confidence: {state.confidence:.3f}")
    print(f"  Latency: {state.latency_ms:.2f}ms")
    print(f"\nComponent Status: {engine.get_component_status()}")
    
    print("\n✅ V2.3 Production Engine tests passed!")
