"""
POMDP Regime Controller
=======================

V2.3 Partially Observable MDP framework for regime-aware decisions.

Core Concept:
The true market regime is hidden (unobservable). We observe noisy signals
(prices, volatility, TDA features) and must maintain belief states about
the underlying regime.

Architecture:
1. Belief State Tracker: Maintains probability distribution over regimes
2. Hidden State Inference: Uses filtering to estimate regime
3. Policy Conditioning: Actions depend on belief state
4. Regime Transition Model: Learned from data

Key Features:
- Bayesian belief updates for regime tracking
- Recurrent architecture for temporal dependencies
- Integration with TDA anomaly features as observations
- Smooth regime transitions (no hard switching)

Research Basis:
- POMDP theory provides principled uncertainty handling
- Hidden Markov Models proven effective for regime detection
- Belief state policies outperform reactive policies

Target Performance:
- Smooth regime transitions (no whipsawing)
- Improved drawdown control during regime shifts
- Better adaptation to changing market conditions
"""

import numpy as np
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using fallback")


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

class MarketRegime(Enum):
    """Market regime states."""
    BULL_QUIET = 0      # Trending up, low volatility
    BULL_VOLATILE = 1   # Trending up, high volatility
    BEAR_QUIET = 2      # Trending down, low volatility
    BEAR_VOLATILE = 3   # Trending down, high volatility (crisis)
    SIDEWAYS = 4        # Range-bound market
    
    @classmethod
    def num_regimes(cls) -> int:
        return len(cls)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class POMDPConfig:
    """Configuration for POMDP Controller."""
    
    # Observation dimensions
    observation_dim: int = 32
    tda_dim: int = 20
    macro_dim: int = 4
    
    # Hidden state
    hidden_dim: int = 64
    belief_dim: int = 32
    n_regimes: int = 5
    
    # Recurrent architecture
    rnn_type: str = 'gru'  # 'gru', 'lstm'
    rnn_layers: int = 2
    
    # Belief dynamics
    belief_smoothing: float = 0.3  # Smooth transitions (0=instant, 1=very slow)
    min_belief: float = 0.01       # Minimum belief probability
    
    # Transition model
    transition_prior: str = 'sticky'  # 'uniform', 'sticky', 'learned'
    sticky_probability: float = 0.95
    
    # Training
    learning_rate: float = 1e-4
    sequence_length: int = 60
    
    # Action space
    action_dim: int = 1
    max_position: float = 0.03
    
    # Regime-specific risk limits
    regime_risk_scales: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,    # BULL_QUIET: Full risk
        1: 0.7,    # BULL_VOLATILE: Reduced risk
        2: 0.5,    # BEAR_QUIET: Conservative
        3: 0.2,    # BEAR_VOLATILE: Very conservative
        4: 0.6,    # SIDEWAYS: Moderate
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# BELIEF STATE TRACKER (Numpy-based)
# =============================================================================

class BeliefStateTracker:
    """
    Maintains belief distribution over hidden market regimes.
    
    Uses Bayesian filtering to update beliefs based on observations.
    """
    
    def __init__(self, config: POMDPConfig):
        self.config = config
        self.n_regimes = config.n_regimes
        
        # Current belief distribution
        self.belief = np.ones(self.n_regimes) / self.n_regimes
        
        # Transition matrix (learned or prior)
        self.transition_matrix = self._init_transition_matrix()
        
        # Observation model parameters (will be learned)
        self.observation_means = np.zeros((self.n_regimes, config.observation_dim))
        self.observation_stds = np.ones((self.n_regimes, config.observation_dim))
        
        # History for training
        self.belief_history = deque(maxlen=1000)
        self.observation_history = deque(maxlen=1000)
        
    def _init_transition_matrix(self) -> np.ndarray:
        """Initialize regime transition matrix."""
        n = self.n_regimes
        
        if self.config.transition_prior == 'uniform':
            return np.ones((n, n)) / n
            
        elif self.config.transition_prior == 'sticky':
            # High probability of staying in same regime
            p_stay = self.config.sticky_probability
            p_leave = (1 - p_stay) / (n - 1)
            matrix = np.full((n, n), p_leave)
            np.fill_diagonal(matrix, p_stay)
            return matrix
            
        else:  # learned - initialize as sticky
            p_stay = 0.9
            p_leave = (1 - p_stay) / (n - 1)
            matrix = np.full((n, n), p_leave)
            np.fill_diagonal(matrix, p_stay)
            return matrix
    
    def reset(self):
        """Reset belief to uniform."""
        self.belief = np.ones(self.n_regimes) / self.n_regimes
        
    def update(self, observation: np.ndarray) -> np.ndarray:
        """
        Update belief given new observation.
        
        Uses Bayes filter:
        1. Predict: belief' = T @ belief
        2. Update: belief = p(obs|regime) * belief' / Z
        
        Args:
            observation: Current observation vector
            
        Returns:
            Updated belief distribution
        """
        # Predict step (transition)
        predicted_belief = self.transition_matrix.T @ self.belief
        
        # Update step (observation likelihood)
        likelihoods = self._observation_likelihood(observation)
        unnormalized = likelihoods * predicted_belief
        
        # Normalize
        normalizer = unnormalized.sum() + 1e-10
        new_belief = unnormalized / normalizer
        
        # Apply minimum belief threshold
        new_belief = np.maximum(new_belief, self.config.min_belief)
        new_belief = new_belief / new_belief.sum()
        
        # Smooth transition
        alpha = self.config.belief_smoothing
        self.belief = alpha * self.belief + (1 - alpha) * new_belief
        self.belief = self.belief / self.belief.sum()
        
        # Store history
        self.belief_history.append(self.belief.copy())
        self.observation_history.append(observation.copy())
        
        return self.belief
    
    def _observation_likelihood(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute likelihood of observation under each regime.
        
        Uses Gaussian emission model.
        """
        likelihoods = np.zeros(self.n_regimes)
        
        for r in range(self.n_regimes):
            diff = observation[:len(self.observation_means[r])] - self.observation_means[r]
            var = self.observation_stds[r] ** 2 + 1e-6
            log_like = -0.5 * np.sum(diff ** 2 / var)
            likelihoods[r] = np.exp(log_like)
            
        # Avoid all zeros
        if likelihoods.sum() < 1e-10:
            likelihoods = np.ones(self.n_regimes)
            
        return likelihoods
    
    def get_most_likely_regime(self) -> int:
        """Get the most likely regime."""
        return int(np.argmax(self.belief))
    
    def get_regime_name(self) -> str:
        """Get name of most likely regime."""
        regime_idx = self.get_most_likely_regime()
        return MarketRegime(regime_idx).name
    
    def get_risk_scale(self) -> float:
        """Get risk scale based on current belief."""
        # Weighted average of regime-specific scales
        risk_scale = 0.0
        for r in range(self.n_regimes):
            risk_scale += self.belief[r] * self.config.regime_risk_scales.get(r, 1.0)
        return float(risk_scale)
    
    def fit_observation_model(self, observations: np.ndarray, labels: np.ndarray):
        """
        Fit observation model from labeled data.
        
        Args:
            observations: [n_samples, obs_dim]
            labels: [n_samples] regime labels
        """
        for r in range(self.n_regimes):
            mask = labels == r
            if mask.sum() > 0:
                regime_obs = observations[mask]
                self.observation_means[r] = regime_obs.mean(axis=0)
                self.observation_stds[r] = regime_obs.std(axis=0) + 1e-6


# =============================================================================
# PYTORCH IMPLEMENTATION
# =============================================================================

if TORCH_AVAILABLE:
    
    class ObservationEncoder(nn.Module):
        """
        Encodes raw observations into latent space.
        
        Combines price features, TDA features, and macro variables.
        """
        
        def __init__(
            self,
            observation_dim: int,
            tda_dim: int,
            macro_dim: int,
            hidden_dim: int
        ):
            super().__init__()
            
            total_dim = observation_dim + tda_dim + macro_dim
            
            self.encoder = nn.Sequential(
                nn.Linear(total_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            
        def forward(
            self, 
            obs: torch.Tensor,
            tda: Optional[torch.Tensor] = None,
            macro: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Encode observations.
            
            Args:
                obs: [batch, seq, obs_dim]
                tda: [batch, seq, tda_dim]
                macro: [batch, seq, macro_dim]
                
            Returns:
                encoded: [batch, seq, hidden_dim]
            """
            inputs = [obs]
            if tda is not None:
                inputs.append(tda)
            if macro is not None:
                inputs.append(macro)
                
            x = torch.cat(inputs, dim=-1)
            return self.encoder(x)


    class BeliefNetwork(nn.Module):
        """
        Neural network for belief state dynamics.
        
        Uses recurrent architecture to track belief over time.
        """
        
        def __init__(self, config: POMDPConfig):
            super().__init__()
            self.config = config
            
            # Observation encoder
            self.obs_encoder = ObservationEncoder(
                observation_dim=config.observation_dim,
                tda_dim=config.tda_dim,
                macro_dim=config.macro_dim,
                hidden_dim=config.hidden_dim
            )
            
            # Recurrent layer for temporal dependencies
            if config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(
                    input_size=config.hidden_dim,
                    hidden_size=config.belief_dim,
                    num_layers=config.rnn_layers,
                    batch_first=True,
                    dropout=0.1 if config.rnn_layers > 1 else 0
                )
            else:  # GRU
                self.rnn = nn.GRU(
                    input_size=config.hidden_dim,
                    hidden_size=config.belief_dim,
                    num_layers=config.rnn_layers,
                    batch_first=True,
                    dropout=0.1 if config.rnn_layers > 1 else 0
                )
            
            # Belief head: outputs regime probabilities
            self.belief_head = nn.Sequential(
                nn.Linear(config.belief_dim, config.belief_dim),
                nn.ReLU(),
                nn.Linear(config.belief_dim, config.n_regimes),
            )
            
            # Transition model: predicts next belief from current
            self.transition_model = nn.Sequential(
                nn.Linear(config.n_regimes, config.n_regimes * 2),
                nn.ReLU(),
                nn.Linear(config.n_regimes * 2, config.n_regimes),
            )
            
        def forward(
            self,
            observations: torch.Tensor,
            tda: Optional[torch.Tensor] = None,
            macro: Optional[torch.Tensor] = None,
            hidden: Optional[Tuple[torch.Tensor, ...]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute belief states for observation sequence.
            
            Args:
                observations: [batch, seq, obs_dim]
                tda: [batch, seq, tda_dim]
                macro: [batch, seq, macro_dim]
                hidden: Initial RNN hidden state
                
            Returns:
                beliefs: [batch, seq, n_regimes] belief probabilities
                hidden: Final RNN hidden state
            """
            # Encode observations
            encoded = self.obs_encoder(observations, tda, macro)
            
            # Process through RNN
            if hidden is None:
                rnn_out, new_hidden = self.rnn(encoded)
            else:
                rnn_out, new_hidden = self.rnn(encoded, hidden)
            
            # Compute belief logits
            belief_logits = self.belief_head(rnn_out)
            
            # Apply softmax for probabilities
            beliefs = F.softmax(belief_logits, dim=-1)
            
            return beliefs, new_hidden
        
        def get_initial_hidden(self, batch_size: int, device: torch.device):
            """Get initial hidden state."""
            if self.config.rnn_type == 'lstm':
                h0 = torch.zeros(
                    self.config.rnn_layers, batch_size, self.config.belief_dim,
                    device=device
                )
                c0 = torch.zeros(
                    self.config.rnn_layers, batch_size, self.config.belief_dim,
                    device=device
                )
                return (h0, c0)
            else:
                return torch.zeros(
                    self.config.rnn_layers, batch_size, self.config.belief_dim,
                    device=device
                )


    class PolicyNetwork(nn.Module):
        """
        Policy conditioned on belief state.
        
        Outputs position sizing based on market state and regime belief.
        """
        
        def __init__(self, config: POMDPConfig):
            super().__init__()
            self.config = config
            
            # Input: observations + belief
            input_dim = config.hidden_dim + config.n_regimes
            
            self.policy = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.action_dim * 2),  # mean + std
            )
            
            # Regime-specific risk scaling
            self.risk_scales = nn.Parameter(
                torch.tensor([
                    config.regime_risk_scales.get(r, 1.0) 
                    for r in range(config.n_regimes)
                ], dtype=torch.float32)
            )
            
        def forward(
            self,
            obs_encoding: torch.Tensor,
            beliefs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute policy output.
            
            Args:
                obs_encoding: [batch, hidden_dim]
                beliefs: [batch, n_regimes]
                
            Returns:
                mean: [batch, action_dim] action mean
                std: [batch, action_dim] action std
            """
            x = torch.cat([obs_encoding, beliefs], dim=-1)
            output = self.policy(x)
            
            mean, log_std = output.chunk(2, dim=-1)
            std = F.softplus(log_std) + 1e-4
            
            # Apply belief-weighted risk scaling
            risk_scale = (beliefs * self.risk_scales).sum(dim=-1, keepdim=True)
            mean = mean * risk_scale * self.config.max_position
            
            return mean, std


    class POMDPController:
        """
        Full POMDP Controller with belief tracking and policy.
        
        Integrates:
        - Observation encoding
        - Belief network for regime inference
        - Belief-conditioned policy
        """
        
        def __init__(self, config: POMDPConfig, device: str = 'cpu'):
            self.config = config
            self.device = torch.device(device)
            
            # Networks
            self.belief_net = BeliefNetwork(config).to(self.device)
            self.policy_net = PolicyNetwork(config).to(self.device)
            
            # Observation encoder for policy (shared architecture)
            self.policy_encoder = ObservationEncoder(
                observation_dim=config.observation_dim,
                tda_dim=config.tda_dim,
                macro_dim=config.macro_dim,
                hidden_dim=config.hidden_dim
            ).to(self.device)
            
            # Optimizers
            self.belief_optimizer = optim.Adam(
                self.belief_net.parameters(), lr=config.learning_rate
            )
            self.policy_optimizer = optim.Adam(
                list(self.policy_net.parameters()) + list(self.policy_encoder.parameters()),
                lr=config.learning_rate
            )
            
            # Tracking
            self.hidden_state = None
            self.current_belief = torch.ones(config.n_regimes, device=self.device) / config.n_regimes
            self.belief_history = []
            
            # Fallback tracker
            self.numpy_tracker = BeliefStateTracker(config)
            
        def reset(self):
            """Reset controller state."""
            self.hidden_state = None
            self.current_belief = torch.ones(
                self.config.n_regimes, device=self.device
            ) / self.config.n_regimes
            self.belief_history = []
            self.numpy_tracker.reset()
            
        @torch.no_grad()
        def update_belief(
            self,
            observation: np.ndarray,
            tda: Optional[np.ndarray] = None,
            macro: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """
            Update belief given new observation.
            
            Args:
                observation: Raw observation
                tda: TDA features
                macro: Macro variables
                
            Returns:
                belief: Updated belief distribution
            """
            self.belief_net.eval()
            
            # Prepare tensors
            obs = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(self.device)
            
            tda_t = None
            if tda is not None:
                tda_t = torch.FloatTensor(tda).unsqueeze(0).unsqueeze(0).to(self.device)
                
            macro_t = None
            if macro is not None:
                macro_t = torch.FloatTensor(macro).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get belief
            beliefs, self.hidden_state = self.belief_net(
                obs, tda_t, macro_t, self.hidden_state
            )
            
            self.current_belief = beliefs[0, -1]  # [n_regimes]
            self.belief_history.append(self.current_belief.cpu().numpy().copy())
            
            return self.current_belief.cpu().numpy()
        
        @torch.no_grad()
        def select_action(
            self,
            observation: np.ndarray,
            tda: Optional[np.ndarray] = None,
            macro: Optional[np.ndarray] = None,
            deterministic: bool = False
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            """
            Select action based on observation and belief.
            
            Args:
                observation: Current observation
                tda: TDA features
                macro: Macro variables
                deterministic: Use mean action
                
            Returns:
                action: Selected action
                info: Additional info (belief, regime, etc.)
            """
            self.policy_net.eval()
            self.policy_encoder.eval()
            
            # Update belief
            belief = self.update_belief(observation, tda, macro)
            
            # Encode observation for policy
            obs = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            inputs = [obs]
            if tda is not None:
                inputs.append(torch.FloatTensor(tda).unsqueeze(0).to(self.device))
            if macro is not None:
                inputs.append(torch.FloatTensor(macro).unsqueeze(0).to(self.device))
            
            x = torch.cat(inputs, dim=-1)
            obs_encoding = self.policy_encoder.encoder(x)
            
            # Get action
            mean, std = self.policy_net(obs_encoding, self.current_belief.unsqueeze(0))
            
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            
            action = torch.clamp(action, 0, self.config.max_position)
            
            # Info
            regime_idx = int(torch.argmax(self.current_belief))
            info = {
                'belief': belief,
                'regime_idx': regime_idx,
                'regime_name': MarketRegime(regime_idx).name,
                'risk_scale': self.get_risk_scale(),
                'action_mean': mean.cpu().numpy().squeeze(),
                'action_std': std.cpu().numpy().squeeze(),
            }
            
            return action.cpu().numpy().squeeze(), info
        
        def get_regime(self) -> MarketRegime:
            """Get most likely regime."""
            return MarketRegime(int(torch.argmax(self.current_belief)))
        
        def get_risk_scale(self) -> float:
            """Get belief-weighted risk scale."""
            risk_scales = torch.tensor([
                self.config.regime_risk_scales.get(r, 1.0)
                for r in range(self.config.n_regimes)
            ], device=self.device)
            return float((self.current_belief * risk_scales).sum())
        
        def train_belief_network(
            self,
            observations: np.ndarray,
            labels: np.ndarray,
            n_epochs: int = 10
        ) -> List[float]:
            """
            Train belief network on labeled regime data.
            
            Args:
                observations: [n_samples, obs_dim]
                labels: [n_samples] regime labels
                n_epochs: Training epochs
                
            Returns:
                losses: Training loss history
            """
            self.belief_net.train()
            
            # Prepare data
            seq_len = self.config.sequence_length
            n_samples = len(observations) - seq_len
            
            losses = []
            
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0
                
                for i in range(0, n_samples, 32):
                    batch_obs = []
                    batch_labels = []
                    
                    for j in range(i, min(i + 32, n_samples)):
                        batch_obs.append(observations[j:j+seq_len])
                        batch_labels.append(labels[j:j+seq_len])
                    
                    batch_size = len(batch_obs)
                    obs = torch.FloatTensor(np.array(batch_obs)).to(self.device)
                    lbls = torch.LongTensor(np.array(batch_labels)).to(self.device)
                    
                    # Create zero-filled tda and macro tensors
                    tda = torch.zeros(batch_size, seq_len, self.config.tda_dim, device=self.device)
                    macro = torch.zeros(batch_size, seq_len, self.config.macro_dim, device=self.device)
                    
                    # Forward pass
                    beliefs, _ = self.belief_net(obs, tda, macro)
                    
                    # Cross entropy loss
                    loss = F.cross_entropy(
                        beliefs.reshape(-1, self.config.n_regimes),
                        lbls.reshape(-1)
                    )
                    
                    # Backward pass
                    self.belief_optimizer.zero_grad()
                    loss.backward()
                    self.belief_optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                avg_loss = epoch_loss / max(n_batches, 1)
                losses.append(avg_loss)
                
            return losses
        
        def save(self, path: str):
            """Save model checkpoint."""
            torch.save({
                'belief_net': self.belief_net.state_dict(),
                'policy_net': self.policy_net.state_dict(),
                'policy_encoder': self.policy_encoder.state_dict(),
                'config': self.config.to_dict(),
            }, path)
            
        def load(self, path: str):
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.belief_net.load_state_dict(checkpoint['belief_net'])
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.policy_encoder.load_state_dict(checkpoint['policy_encoder'])


# =============================================================================
# FALLBACK IMPLEMENTATION
# =============================================================================

class FallbackPOMDPController:
    """Simple fallback without PyTorch."""
    
    def __init__(self, config: POMDPConfig):
        self.config = config
        self.tracker = BeliefStateTracker(config)
        
    def reset(self):
        """Reset controller."""
        self.tracker.reset()
        
    def update_belief(
        self,
        observation: np.ndarray,
        tda: Optional[np.ndarray] = None,
        macro: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Update belief."""
        return self.tracker.update(observation)
    
    def select_action(
        self,
        observation: np.ndarray,
        tda: Optional[np.ndarray] = None,
        macro: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action."""
        belief = self.update_belief(observation, tda, macro)
        
        # Simple action based on regime
        risk_scale = self.tracker.get_risk_scale()
        action = np.array([self.config.max_position * risk_scale])
        
        info = {
            'belief': belief,
            'regime_idx': self.tracker.get_most_likely_regime(),
            'regime_name': self.tracker.get_regime_name(),
            'risk_scale': risk_scale,
        }
        
        return action, info
    
    def get_regime(self) -> MarketRegime:
        """Get most likely regime."""
        return MarketRegime(self.tracker.get_most_likely_regime())
    
    def get_risk_scale(self) -> float:
        """Get risk scale."""
        return self.tracker.get_risk_scale()


# Use appropriate implementation
if not TORCH_AVAILABLE:
    POMDPController = FallbackPOMDPController


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = POMDPConfig(
        observation_dim=20,
        tda_dim=10,
        macro_dim=4,
        hidden_dim=32,
        belief_dim=16,
        n_regimes=5,
    )
    
    print("Testing BeliefStateTracker...")
    tracker = BeliefStateTracker(config)
    
    # Simulate observations
    for i in range(100):
        obs = np.random.randn(config.observation_dim)
        belief = tracker.update(obs)
        
    print(f"Final belief: {tracker.belief}")
    print(f"Most likely regime: {tracker.get_regime_name()}")
    print(f"Risk scale: {tracker.get_risk_scale():.3f}")
    
    if TORCH_AVAILABLE:
        print("\nTesting PyTorch POMDPController...")
        controller = POMDPController(config, device='cpu')
        
        # Test action selection
        for i in range(50):
            obs = np.random.randn(config.observation_dim)
            tda = np.random.randn(config.tda_dim)
            macro = np.random.randn(config.macro_dim)
            
            action, info = controller.select_action(obs, tda, macro)
            
            if i % 10 == 0:
                print(f"Step {i}: action={action:.4f}, "
                      f"regime={info['regime_name']}, "
                      f"risk_scale={info['risk_scale']:.3f}")
        
        # Test belief training
        print("\nTesting belief network training...")
        n_samples = 500
        observations = np.random.randn(n_samples, config.observation_dim)
        labels = np.random.randint(0, config.n_regimes, n_samples)
        
        losses = controller.train_belief_network(observations, labels, n_epochs=3)
        print(f"Training losses: {losses}")
        
        print("\n✅ POMDP Controller tests passed!")
    else:
        print("\nTesting Fallback POMDPController...")
        controller = FallbackPOMDPController(config)
        
        obs = np.random.randn(config.observation_dim)
        action, info = controller.select_action(obs)
        print(f"Action: {action}, Regime: {info['regime_name']}")
        print("✅ Fallback tests passed!")
