"""
Dueling Soft Actor-Critic with Distributional RL
=================================================

V2.3 Enhanced SAC combining multiple advanced RL techniques:

1. Dueling Networks: Separate state value and advantage streams
2. Distributional RL: Model full return distribution (not just expectation)
3. Prioritized Experience Replay: Focus on important transitions
4. Double Q-Learning: Reduce overestimation bias
5. Enhanced Entropy Regularization: Adaptive temperature

Key Features:
- Dueling architecture separates V(s) and A(s,a) for better generalization
- Quantile regression for distributional critic (IQN-style)
- Risk-sensitive policy using CVaR from return distribution
- Integration with PrioritizedReplayBuffer

Research Basis:
- Dueling DQN: +40% improvement on Atari (Wang et al. 2016)
- Distributional RL captures tail risk crucial for trading
- SAC entropy regularization improves exploration

Target Performance:
- Improved sample efficiency vs vanilla SAC
- Better tail risk management via CVaR optimization
- Sharpe improvement: +0.3-0.5 over baseline
"""

import numpy as np
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using fallback")

# Import PER buffer
try:
    from src.agents.prioritized_replay_buffer import (
        PrioritizedReplayBuffer, 
        TorchPrioritizedReplayBuffer,
        PERConfig
    )
    PER_AVAILABLE = True
except ImportError:
    PER_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DuelingSACConfig:
    """Configuration for Dueling SAC."""
    
    # Dimensions
    state_dim: int = 32
    action_dim: int = 1
    hidden_dims: Tuple[int, ...] = (256, 256)
    
    # Dueling architecture
    value_hidden_dim: int = 128
    advantage_hidden_dim: int = 128
    
    # Distributional RL
    n_quantiles: int = 32          # Number of quantile samples
    risk_distortion: str = 'cvar'  # 'neutral', 'cvar', 'wang'
    cvar_alpha: float = 0.25       # CVaR confidence level
    
    # SAC hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    alpha: float = 0.2             # Entropy coefficient
    auto_entropy: bool = True
    target_entropy: Optional[float] = None  # Auto-computed if None
    
    # Replay buffer
    buffer_size: int = 1_000_000
    min_buffer_size: int = 1000
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta: float = 0.4
    
    # Training
    gradient_clip: float = 1.0
    updates_per_step: int = 1
    
    # Position constraints
    max_position: float = 0.03
    min_position: float = 0.0
    
    # Reward shaping
    reward_scale: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# FALLBACK IMPLEMENTATION
# =============================================================================

class FallbackDuelingSAC:
    """Simple fallback without PyTorch."""
    
    def __init__(self, config: DuelingSACConfig):
        self.config = config
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Simple random action."""
        action = np.random.uniform(
            self.config.min_position,
            self.config.max_position,
            size=(self.config.action_dim,)
        )
        return action
    
    def train_step(self, batch):
        """No-op training."""
        return {'loss': 0.0}


# =============================================================================
# PYTORCH IMPLEMENTATION
# =============================================================================

if TORCH_AVAILABLE:
    
    class DuelingCritic(nn.Module):
        """
        Dueling Q-Network with distributional output.
        
        Decomposes Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        
        For distributional RL, outputs quantile values instead of
        point estimates.
        """
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Tuple[int, ...],
            value_hidden_dim: int,
            advantage_hidden_dim: int,
            n_quantiles: int = 32
        ):
            super().__init__()
            
            self.n_quantiles = n_quantiles
            
            # Shared feature extractor
            layers = []
            input_dim = state_dim + action_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            self.features = nn.Sequential(*layers)
            
            # Value stream: V(s) - state value
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dims[-1], value_hidden_dim),
                nn.ReLU(),
                nn.Linear(value_hidden_dim, n_quantiles),  # Quantile values
            )
            
            # Advantage stream: A(s,a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dims[-1], advantage_hidden_dim),
                nn.ReLU(),
                nn.Linear(advantage_hidden_dim, n_quantiles),
            )
            
        def forward(
            self, 
            state: torch.Tensor, 
            action: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute Q-value quantiles.
            
            Args:
                state: [batch, state_dim]
                action: [batch, action_dim]
                
            Returns:
                quantiles: [batch, n_quantiles]
            """
            x = torch.cat([state, action], dim=-1)
            features = self.features(x)
            
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine: Q = V + A - mean(A)
            q_quantiles = value + advantage - advantage.mean(dim=-1, keepdim=True)
            
            return q_quantiles
        
        def get_expected_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """Get expected Q-value (mean of quantiles)."""
            quantiles = self.forward(state, action)
            return quantiles.mean(dim=-1)


    class SquashedGaussianActor(nn.Module):
        """
        Stochastic actor with squashed Gaussian output.
        
        Outputs tanh-squashed actions for bounded action space.
        Uses reparameterization trick for gradient flow.
        """
        
        LOG_STD_MIN = -20
        LOG_STD_MAX = 2
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Tuple[int, ...],
            action_scale: float = 1.0
        ):
            super().__init__()
            
            self.action_dim = action_dim
            self.action_scale = action_scale
            
            # Feature layers
            layers = []
            input_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            self.features = nn.Sequential(*layers)
            
            # Mean and log_std heads
            self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
            
        def forward(
            self, 
            state: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Get action distribution parameters.
            
            Returns:
                mean, log_std
            """
            features = self.features(state)
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            
            return mean, log_std
        
        def sample(
            self, 
            state: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Sample action with log probability.
            
            Returns:
                action: Squashed action
                log_prob: Log probability of action
            """
            mean, log_std = self.forward(state)
            
            if deterministic:
                action = torch.tanh(mean) * self.action_scale
                return action, torch.zeros_like(action)
            
            std = log_std.exp()
            normal = Normal(mean, std)
            
            # Reparameterization trick
            x = normal.rsample()
            action = torch.tanh(x) * self.action_scale
            
            # Log probability with tanh correction
            log_prob = normal.log_prob(x)
            log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return action, log_prob


    class DuelingSAC:
        """
        Dueling Soft Actor-Critic with Distributional RL.
        
        Combines:
        - Dueling networks for V/A decomposition
        - Distributional critics for risk-aware learning
        - Entropy-regularized policy optimization
        - Prioritized experience replay
        """
        
        def __init__(self, config: DuelingSACConfig, device: str = 'cpu'):
            self.config = config
            self.device = torch.device(device)
            
            # Action scale (maps tanh output to position range)
            self.action_scale = config.max_position
            
            # Actor network
            self.actor = SquashedGaussianActor(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dims=config.hidden_dims,
                action_scale=self.action_scale
            ).to(self.device)
            
            # Twin dueling critics
            self.critic1 = DuelingCritic(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dims=config.hidden_dims,
                value_hidden_dim=config.value_hidden_dim,
                advantage_hidden_dim=config.advantage_hidden_dim,
                n_quantiles=config.n_quantiles
            ).to(self.device)
            
            self.critic2 = DuelingCritic(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dims=config.hidden_dims,
                value_hidden_dim=config.value_hidden_dim,
                advantage_hidden_dim=config.advantage_hidden_dim,
                n_quantiles=config.n_quantiles
            ).to(self.device)
            
            # Target critics
            self.critic1_target = DuelingCritic(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dims=config.hidden_dims,
                value_hidden_dim=config.value_hidden_dim,
                advantage_hidden_dim=config.advantage_hidden_dim,
                n_quantiles=config.n_quantiles
            ).to(self.device)
            
            self.critic2_target = DuelingCritic(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dims=config.hidden_dims,
                value_hidden_dim=config.value_hidden_dim,
                advantage_hidden_dim=config.advantage_hidden_dim,
                n_quantiles=config.n_quantiles
            ).to(self.device)
            
            # Initialize targets
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            
            # Optimizers
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=config.learning_rate
            )
            self.critic1_optimizer = optim.Adam(
                self.critic1.parameters(), lr=config.learning_rate
            )
            self.critic2_optimizer = optim.Adam(
                self.critic2.parameters(), lr=config.learning_rate
            )
            
            # Entropy coefficient
            if config.auto_entropy:
                self.target_entropy = config.target_entropy or -config.action_dim
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
            else:
                self.log_alpha = torch.tensor(np.log(config.alpha), device=self.device)
                
            # Replay buffer
            if config.use_per and PER_AVAILABLE:
                per_config = PERConfig(
                    capacity=config.buffer_size,
                    alpha=config.per_alpha,
                    beta=config.per_beta
                )
                self.buffer = TorchPrioritizedReplayBuffer(per_config, device=device)
            else:
                # Use simple deque-based buffer
                from collections import deque
                self.buffer = deque(maxlen=config.buffer_size)
                self._simple_buffer = True
                
            # Quantile tau values for distributional RL
            self.tau_values = torch.linspace(
                0.5 / config.n_quantiles,
                1 - 0.5 / config.n_quantiles,
                config.n_quantiles,
                device=self.device
            )
            
            # Risk distortion weights
            self.risk_weights = self._compute_risk_weights()
            
            # Training stats
            self.total_steps = 0
            self.training_stats = []
            
        def _compute_risk_weights(self) -> torch.Tensor:
            """Compute risk distortion weights based on config."""
            if self.config.risk_distortion == 'neutral':
                return torch.ones(self.config.n_quantiles, device=self.device) / self.config.n_quantiles
                
            elif self.config.risk_distortion == 'cvar':
                # CVaR: Focus on worst α quantiles
                alpha = self.config.cvar_alpha
                weights = torch.zeros(self.config.n_quantiles, device=self.device)
                n_cvar = int(alpha * self.config.n_quantiles)
                weights[:n_cvar] = 1.0 / n_cvar
                return weights
                
            elif self.config.risk_distortion == 'wang':
                # Wang transform for risk aversion
                eta = 0.75  # Risk aversion parameter
                taus = self.tau_values.cpu().numpy()
                from scipy.stats import norm
                distorted = norm.cdf(norm.ppf(taus) + eta)
                weights = np.diff(np.concatenate([[0], distorted]))
                return torch.tensor(weights, device=self.device, dtype=torch.float32)
            else:
                return torch.ones(self.config.n_quantiles, device=self.device) / self.config.n_quantiles
        
        @property
        def alpha(self) -> torch.Tensor:
            """Current entropy coefficient."""
            return self.log_alpha.exp()
        
        def select_action(
            self, 
            state: np.ndarray, 
            deterministic: bool = False
        ) -> np.ndarray:
            """
            Select action for given state.
            
            Args:
                state: Current state
                deterministic: If True, use mean action
                
            Returns:
                action: Selected action
            """
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _ = self.actor.sample(state_tensor, deterministic)
                
            return action.cpu().numpy().squeeze(0)
        
        def store_transition(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool
        ):
            """Store transition in replay buffer."""
            if hasattr(self, '_simple_buffer') and self._simple_buffer:
                self.buffer.append((state, action, reward, next_state, done))
            else:
                self.buffer.add(state, action, reward, next_state, done)
            
        def _sample_simple_buffer(self, batch_size: int):
            """Sample from simple deque buffer."""
            import random
            batch = random.sample(list(self.buffer), batch_size)
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            rewards = np.array([t[2] for t in batch]).reshape(-1, 1)
            next_states = np.array([t[3] for t in batch])
            dones = np.array([t[4] for t in batch], dtype=np.float32).reshape(-1, 1)
            
            return {
                'states': torch.FloatTensor(states).to(self.device),
                'actions': torch.FloatTensor(actions).to(self.device),
                'rewards': torch.FloatTensor(rewards).to(self.device),
                'next_states': torch.FloatTensor(next_states).to(self.device),
                'dones': torch.FloatTensor(dones).to(self.device),
            }
        
        def train_step(self) -> Dict[str, float]:
            """
            Perform single training step.
            
            Returns:
                Dictionary of training metrics
            """
            if len(self.buffer) < self.config.min_buffer_size:
                return {'buffer_size': len(self.buffer)}
                
            # Sample batch based on buffer type
            if hasattr(self, '_simple_buffer') and self._simple_buffer:
                batch = self._sample_simple_buffer(self.config.batch_size)
                weights = torch.ones(self.config.batch_size, device=self.device)
                indices = None
            elif hasattr(self.buffer, 'sample_torch'):
                batch, weights, indices = self.buffer.sample_torch(self.config.batch_size)
            else:
                batch_np, weights, _ = self.buffer.sample(self.config.batch_size)
                batch = {
                    k: torch.FloatTensor(v).to(self.device) 
                    for k, v in batch_np.items()
                }
                if batch['rewards'].dim() == 1:
                    batch['rewards'] = batch['rewards'].unsqueeze(-1)
                if batch['dones'].dim() == 1:
                    batch['dones'] = batch['dones'].unsqueeze(-1)
                weights = torch.FloatTensor(weights).to(self.device)
                indices = None
            
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards'] * self.config.reward_scale
            next_states = batch['next_states']
            dones = batch['dones']
            
            # Compute critic loss (distributional)
            critic_loss, td_errors = self._compute_critic_loss(
                states, actions, rewards, next_states, dones, weights
            )
            
            # Update critics
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.gradient_clip)
                nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.gradient_clip)
                
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
            
            # Update priorities in PER
            if indices is not None and hasattr(self.buffer, 'update_priorities'):
                priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
                self.buffer.update_priorities(indices, priorities)
            
            # Compute actor loss
            actor_loss, log_probs = self._compute_actor_loss(states)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
                
            self.actor_optimizer.step()
            
            # Update entropy coefficient
            alpha_loss = torch.tensor(0.0)
            if self.config.auto_entropy:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            
            # Soft update targets
            self._soft_update()
            
            self.total_steps += 1
            
            metrics = {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha': self.alpha.item(),
                'alpha_loss': alpha_loss.item(),
                'mean_q': td_errors.mean().item(),
                'total_steps': self.total_steps,
            }
            
            self.training_stats.append(metrics)
            
            return metrics
        
        def _compute_critic_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
            weights: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute distributional critic loss."""
            
            with torch.no_grad():
                # Sample next actions
                next_actions, next_log_probs = self.actor.sample(next_states)
                
                # Get target Q quantiles
                target_q1 = self.critic1_target(next_states, next_actions)
                target_q2 = self.critic2_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                
                # Compute target with entropy bonus
                target_q = target_q - self.alpha * next_log_probs
                
                # Bellman target for each quantile
                target_quantiles = rewards + (1 - dones) * self.config.gamma * target_q
            
            # Current Q quantiles
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            # Quantile Huber loss
            def quantile_huber_loss(pred, target, taus):
                # pred: [batch, n_quantiles]
                # target: [batch, n_quantiles]
                errors = target.unsqueeze(-1) - pred.unsqueeze(-2)  # [batch, n_quantiles, n_quantiles]
                huber = torch.where(
                    errors.abs() <= 1.0,
                    0.5 * errors.pow(2),
                    errors.abs() - 0.5
                )
                
                # Quantile weights
                taus_expanded = taus.view(1, 1, -1)
                quantile_weights = torch.abs(taus_expanded - (errors < 0).float())
                
                loss = (quantile_weights * huber).mean(dim=-1).sum(dim=-1)
                return loss
            
            loss1 = quantile_huber_loss(current_q1, target_quantiles, self.tau_values)
            loss2 = quantile_huber_loss(current_q2, target_quantiles, self.tau_values)
            
            # Apply importance sampling weights
            critic_loss = (weights * (loss1 + loss2)).mean()
            
            # TD errors for priority updates
            with torch.no_grad():
                td_errors = (target_quantiles.mean(dim=-1) - current_q1.mean(dim=-1))
            
            return critic_loss, td_errors
        
        def _compute_actor_loss(
            self, 
            states: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute actor loss with risk-aware objective."""
            
            actions, log_probs = self.actor.sample(states)
            
            # Get Q-values with risk distortion
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            q_min = torch.min(q1, q2)
            
            # Risk-weighted expected Q
            if self.config.risk_distortion != 'neutral':
                q_risk = (q_min * self.risk_weights).sum(dim=-1, keepdim=True)
            else:
                q_risk = q_min.mean(dim=-1, keepdim=True)
            
            # Actor loss: maximize Q - alpha * entropy
            actor_loss = (self.alpha * log_probs - q_risk).mean()
            
            return actor_loss, log_probs
        
        def _soft_update(self):
            """Soft update target networks."""
            tau = self.config.tau
            
            for target_param, param in zip(
                self.critic1_target.parameters(), 
                self.critic1.parameters()
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            for target_param, param in zip(
                self.critic2_target.parameters(), 
                self.critic2.parameters()
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        def save(self, path: str):
            """Save model checkpoint."""
            torch.save({
                'actor': self.actor.state_dict(),
                'critic1': self.critic1.state_dict(),
                'critic2': self.critic2.state_dict(),
                'critic1_target': self.critic1_target.state_dict(),
                'critic2_target': self.critic2_target.state_dict(),
                'log_alpha': self.log_alpha,
                'config': self.config.to_dict(),
                'total_steps': self.total_steps,
            }, path)
            
        def load(self, path: str):
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.critic1_target.load_state_dict(checkpoint['critic1_target'])
            self.critic2_target.load_state_dict(checkpoint['critic2_target'])
            self.log_alpha = checkpoint['log_alpha']
            self.total_steps = checkpoint.get('total_steps', 0)
        
        def get_statistics(self) -> Dict[str, Any]:
            """Get training statistics."""
            if not self.training_stats:
                return {}
                
            recent = self.training_stats[-100:]
            return {
                'total_steps': self.total_steps,
                'buffer_size': len(self.buffer),
                'mean_critic_loss': np.mean([s['critic_loss'] for s in recent]),
                'mean_actor_loss': np.mean([s['actor_loss'] for s in recent]),
                'alpha': self.alpha.item(),
            }


# Use appropriate implementation
if TORCH_AVAILABLE:
    pass  # PyTorch classes defined above
else:
    DuelingSAC = FallbackDuelingSAC


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = DuelingSACConfig(
        state_dim=32,
        action_dim=1,
        hidden_dims=(128, 128),
        n_quantiles=16,
        risk_distortion='cvar',
        cvar_alpha=0.25,
        batch_size=64,
        min_buffer_size=100,
    )
    
    if TORCH_AVAILABLE:
        print("Testing Dueling SAC...")
        agent = DuelingSAC(config, device='cpu')
        
        # Fill buffer with random transitions
        state_dim = config.state_dim
        for i in range(200):
            state = np.random.randn(state_dim)
            action = agent.select_action(state)
            reward = np.random.randn() * 0.01
            next_state = np.random.randn(state_dim)
            done = np.random.rand() < 0.05
            
            agent.store_transition(state, action, reward, next_state, done)
        
        print(f"Buffer size: {len(agent.buffer)}")
        
        # Training steps
        for step in range(50):
            metrics = agent.train_step()
            if step % 10 == 0:
                print(f"Step {step}: critic_loss={metrics.get('critic_loss', 0):.4f}, "
                      f"actor_loss={metrics.get('actor_loss', 0):.4f}, "
                      f"alpha={metrics.get('alpha', 0):.4f}")
        
        # Test action selection
        state = np.random.randn(state_dim)
        action = agent.select_action(state, deterministic=True)
        print(f"\nDeterministic action: {action}")
        
        action_stoch = agent.select_action(state, deterministic=False)
        print(f"Stochastic action: {action_stoch}")
        
        print(f"\nStatistics: {agent.get_statistics()}")
        print("\n✅ Dueling SAC tests passed!")
    else:
        print("PyTorch not available - testing fallback")
        agent = FallbackDuelingSAC(config)
        action = agent.select_action(np.random.randn(32))
        print(f"Fallback action: {action}")
        print("✅ Fallback tests passed!")
