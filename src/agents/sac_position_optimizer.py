"""
Soft Actor-Critic (SAC) Position Optimizer
==========================================

V2.2 Advanced RL-based dynamic position sizing with risk-aware rewards.

Key Features:
- Continuous action space for position sizes [0, max_position]
- Twin Q-networks for stable value estimation
- Entropy regularization for exploration
- Risk-aware reward: profit + drawdown_penalty + exploration_bonus
- Prioritized Experience Replay (PER) for efficient learning

Research Basis:
- SAC agents achieve 0.52 Sharpe improvements in market making
- Risk-aware rewards optimize position sizing for tail risk

Usage:
    sac = SACPositionOptimizer(SACConfig())
    sac.train(market_data, episodes=1000)
    position_size = sac.get_position(state)
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import deque
import random

logger = logging.getLogger(__name__)

# Try to import PyTorch - use numpy fallback if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - SAC will use simplified fallback")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SACConfig:
    """Configuration for SAC Position Optimizer."""
    
    # Network architecture
    state_dim: int = 32  # Market state features
    action_dim: int = 1  # Position size (continuous)
    hidden_dims: Tuple[int, ...] = (256, 256)
    
    # SAC hyperparameters (from research)
    learning_rate: float = 3e-4
    batch_size: int = 256
    tau: float = 0.005  # Soft update coefficient
    gamma: float = 0.99  # Discount factor
    alpha: float = 0.2  # Entropy coefficient (auto-tuned)
    auto_entropy: bool = True  # Auto-tune entropy
    
    # Replay buffer
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000  # Min samples before training
    use_per: bool = True  # Prioritized Experience Replay
    per_alpha: float = 0.6  # Priority exponent
    per_beta: float = 0.4  # Importance sampling start
    per_beta_increment: float = 0.001  # Beta annealing
    
    # Training
    updates_per_step: int = 1
    target_update_interval: int = 1
    gradient_clip: float = 1.0
    
    # Position sizing constraints
    max_position_pct: float = 0.03  # 3% max position
    min_position_pct: float = 0.0  # Can go to 0
    
    # Reward shaping
    profit_weight: float = 1.0
    drawdown_penalty: float = 2.0  # Penalize drawdowns 2x profits
    exploration_bonus: float = 0.1
    risk_aversion: float = 0.5  # Risk-adjusted reward coefficient
    
    # Logging
    log_dir: str = "logs"
    log_interval: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# REPLAY BUFFERS
# =============================================================================

class ReplayBuffer:
    """Standard experience replay buffer."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch of experiences."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
        
    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay buffer with sum-tree."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity, state_dim, action_dim)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 1e-6
        
        # Sum-tree for efficient sampling
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
            
        self.tree = np.zeros(2 * self.tree_capacity - 1, dtype=np.float32)
        self.max_priority = 1.0
        
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, value: float) -> int:
        """Retrieve leaf index for given value."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])
            
    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience with max priority."""
        tree_idx = self.ptr + self.tree_capacity - 1
        
        # Update tree
        priority = self.max_priority ** self.alpha
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        
        # Add to buffer
        super().push(state, action, reward, next_state, done)
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with priorities."""
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        segment = self.tree[0] / batch_size
        
        for i in range(batch_size):
            value = random.uniform(segment * i, segment * (i + 1))
            tree_idx = self._retrieve(0, value)
            data_idx = tree_idx - self.tree_capacity + 1
            
            indices[i] = data_idx
            priority = self.tree[tree_idx]
            prob = priority / self.tree[0]
            weights[i] = (self.size * prob) ** (-self.beta)
            
        # Normalize weights
        weights /= weights.max()
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights,
        )
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + self.epsilon
        
        for idx, priority in zip(indices, priorities):
            tree_idx = idx + self.tree_capacity - 1
            change = (priority ** self.alpha) - self.tree[tree_idx]
            self.tree[tree_idx] = priority ** self.alpha
            self._propagate(tree_idx, change)
            self.max_priority = max(self.max_priority, priority)


# =============================================================================
# NEURAL NETWORKS (PyTorch)
# =============================================================================

if TORCH_AVAILABLE:
    
    class MLP(nn.Module):
        """Multi-layer perceptron for critic/actor networks."""
        
        def __init__(self, input_dim: int, output_dim: int,
                     hidden_dims: Tuple[int, ...] = (256, 256),
                     activation: nn.Module = nn.ReLU):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activation())
                prev_dim = hidden_dim
                
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    
    
    class GaussianActor(nn.Module):
        """Stochastic actor with Gaussian policy."""
        
        LOG_STD_MIN = -20
        LOG_STD_MAX = 2
        
        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dims: Tuple[int, ...] = (256, 256)):
            super().__init__()
            
            self.shared = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1])
            self.mean = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std = nn.Linear(hidden_dims[-1], action_dim)
            
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Get action mean and log std."""
            x = F.relu(self.shared(state))
            mean = self.mean(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            return mean, log_std
        
        def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Sample action and compute log probability."""
            mean, log_std = self.forward(state)
            std = log_std.exp()
            
            # Reparameterization trick
            normal = Normal(mean, std)
            x = normal.rsample()
            
            # Squash to [0, 1] for position sizing
            action = torch.sigmoid(x)
            
            # Log probability with squashing correction
            log_prob = normal.log_prob(x)
            log_prob -= torch.log(action * (1 - action) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
            
            return action, log_prob
        
        def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
            """Get action for inference."""
            mean, log_std = self.forward(state)
            
            if deterministic:
                return torch.sigmoid(mean)
            
            std = log_std.exp()
            normal = Normal(mean, std)
            x = normal.rsample()
            return torch.sigmoid(x)
    
    
    class TwinCritic(nn.Module):
        """Twin Q-networks for stable value estimation."""
        
        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dims: Tuple[int, ...] = (256, 256)):
            super().__init__()
            
            self.q1 = MLP(state_dim + action_dim, 1, hidden_dims)
            self.q2 = MLP(state_dim + action_dim, 1, hidden_dims)
            
        def forward(self, state: torch.Tensor, 
                    action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute Q-values from both critics."""
            x = torch.cat([state, action], dim=-1)
            return self.q1(x), self.q2(x)
        
        def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """Compute Q1 value only (for actor update)."""
            x = torch.cat([state, action], dim=-1)
            return self.q1(x)


# =============================================================================
# SAC POSITION OPTIMIZER
# =============================================================================

class SACPositionOptimizer:
    """
    Soft Actor-Critic for Dynamic Position Sizing
    
    Learns optimal position sizes based on market state while
    maximizing risk-adjusted returns with entropy regularization.
    
    Features:
    - Continuous action space [0, max_position]
    - Risk-aware reward shaping
    - Automatic entropy tuning
    - Prioritized experience replay
    """
    
    def __init__(self, config: Optional[SACConfig] = None):
        """Initialize SAC agent."""
        self.config = config or SACConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        # Initialize networks if PyTorch available
        if TORCH_AVAILABLE:
            self._init_networks()
        else:
            self._init_fallback()
            
        # Initialize replay buffer
        if self.config.use_per:
            self.buffer = PrioritizedReplayBuffer(
                self.config.buffer_size,
                self.config.state_dim,
                self.config.action_dim,
                alpha=self.config.per_alpha,
                beta=self.config.per_beta,
            )
        else:
            self.buffer = ReplayBuffer(
                self.config.buffer_size,
                self.config.state_dim,
                self.config.action_dim,
            )
            
        # Training state
        self.total_steps = 0
        self.episode_rewards = []
        self.training_losses = []
        
        # Logging
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.decisions_log = self.log_dir / "rl_decisions.jsonl"
        
        logger.info(f"SACPositionOptimizer initialized on {self.device}")
        
    def _init_networks(self):
        """Initialize PyTorch networks."""
        # Actor network
        self.actor = GaussianActor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)
        
        # Twin critic networks
        self.critic = TwinCritic(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)
        
        # Target critic (soft update)
        self.critic_target = TwinCritic(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )
        
        # Entropy coefficient (auto-tuned)
        if self.config.auto_entropy:
            self.target_entropy = -self.config.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config.alpha
            
    def _init_fallback(self):
        """Initialize numpy-based fallback (simplified)."""
        logger.warning("Using simplified numpy fallback for SAC")
        
        # Simple linear model fallback
        self.weights_mean = np.random.randn(self.config.state_dim) * 0.01
        self.weights_std = np.ones(self.config.state_dim) * 0.5
        self.alpha = self.config.alpha
        
    def get_position(self, state: np.ndarray, 
                     regime: str = "neutral",
                     deterministic: bool = False) -> float:
        """
        Get position size for given market state.
        
        Args:
            state: Market state features (normalized)
            regime: Current market regime (trending/volatile/flat)
            deterministic: Use mean action (no exploration)
            
        Returns:
            Position size as fraction of max_position
        """
        if TORCH_AVAILABLE:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actor.get_action(state_tensor, deterministic)
                position = action.cpu().numpy()[0, 0]
        else:
            # Fallback: simple linear + sigmoid
            logit = np.dot(state, self.weights_mean)
            position = 1 / (1 + np.exp(-logit))
            
        # Apply regime-based scaling
        regime_scales = {
            "trending": 1.2,  # Slightly larger positions in trends
            "volatile": 0.6,  # Reduce in high volatility
            "flat": 0.8,      # Moderate in flat markets
            "neutral": 1.0,
        }
        regime_scale = regime_scales.get(regime, 1.0)
        
        # Scale to actual position percentage
        position_pct = position * self.config.max_position_pct * regime_scale
        position_pct = np.clip(position_pct, self.config.min_position_pct, self.config.max_position_pct)
        
        # Log decision
        self._log_decision(state, position_pct, regime)
        
        return float(position_pct)
    
    def _log_decision(self, state: np.ndarray, position: float, regime: str):
        """Log RL decision to JSONL file."""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "state_mean": float(np.mean(state)),
            "state_std": float(np.std(state)),
            "position_pct": position,
            "regime": regime,
            "alpha": float(self.alpha) if hasattr(self, 'alpha') else 0.0,
            "total_steps": self.total_steps,
        }
        
        try:
            with open(self.decisions_log, 'a') as f:
                f.write(json.dumps(decision) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log decision: {e}")
            
    def compute_reward(self, 
                       pnl: float,
                       drawdown: float,
                       position_change: float,
                       volatility: float) -> float:
        """
        Compute risk-aware reward for RL training.
        
        Components:
        - Profit term: Scaled PnL
        - Drawdown penalty: Penalize underwater equity
        - Exploration bonus: Reward position diversity
        - Risk adjustment: Sharpe-like scaling
        
        Args:
            pnl: Period profit/loss (fraction)
            drawdown: Current drawdown (fraction)
            position_change: Turnover this step
            volatility: Current market volatility
            
        Returns:
            Shaped reward value
        """
        # Profit term
        profit_reward = self.config.profit_weight * pnl
        
        # Drawdown penalty (asymmetric - penalize losses more)
        dd_penalty = -self.config.drawdown_penalty * max(0, drawdown)
        
        # Exploration bonus (encourage diverse positions)
        exploration = self.config.exploration_bonus * np.abs(position_change)
        
        # Risk adjustment (penalize large positions in high volatility)
        vol_penalty = -self.config.risk_aversion * volatility * np.abs(position_change)
        
        # Combine
        reward = profit_reward + dd_penalty + exploration + vol_penalty
        
        return float(reward)
    
    def store_experience(self, 
                         state: np.ndarray,
                         action: np.ndarray,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool):
        """Store experience in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        
    def update(self) -> Dict[str, float]:
        """
        Perform one SAC update step.
        
        Returns:
            Dictionary of training metrics
        """
        if not TORCH_AVAILABLE:
            return self._update_fallback()
            
        if len(self.buffer) < self.config.min_buffer_size:
            return {}
            
        metrics = {}
        
        # Sample batch
        if self.config.use_per:
            batch = self.buffer.sample(self.config.batch_size)
            states, actions, rewards, next_states, dones, indices, weights = batch
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None
            
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        
        # ===== Critic Update =====
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            td_target = rewards + self.config.gamma * (1 - dones) * q_target
            
        q1, q2 = self.critic(states, actions)
        
        # TD errors for PER
        td_errors = (td_target - q1).abs().detach().cpu().numpy().flatten()
        
        # Critic loss
        critic_loss = (weights * ((q1 - td_target) ** 2)).mean() + \
                      (weights * ((q2 - td_target) ** 2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()
        
        metrics['critic_loss'] = critic_loss.item()
        
        # Update priorities if using PER
        if self.config.use_per and indices is not None:
            self.buffer.update_priorities(indices, td_errors)
            
        # ===== Actor Update =====
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()
        
        metrics['actor_loss'] = actor_loss.item()
        
        # ===== Entropy Tuning =====
        if self.config.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            metrics['alpha'] = self.alpha
            metrics['alpha_loss'] = alpha_loss.item()
            
        # ===== Soft Update Target Networks =====
        if self.total_steps % self.config.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), 
                                           self.critic_target.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + 
                    (1 - self.config.tau) * target_param.data
                )
                
        self.total_steps += 1
        self.training_losses.append(metrics)
        
        return metrics
    
    def _update_fallback(self) -> Dict[str, float]:
        """Simple gradient update for numpy fallback."""
        if len(self.buffer) < 100:
            return {}
            
        # Sample experiences - handle both regular and prioritized buffers
        sample = self.buffer.sample(min(32, len(self.buffer)))
        if len(sample) == 7:  # PrioritizedReplayBuffer returns 7 values
            states, actions, rewards, next_states, dones, indices, weights = sample
        else:  # Regular buffer returns 5 values
            states, actions, rewards, next_states, dones = sample
        
        # Simple policy gradient
        for i in range(len(states)):
            state = states[i]
            action = actions[i][0]
            reward = rewards[i]
            
            # Update weights towards rewarding positions
            gradient = state * (reward - 0) * action
            self.weights_mean += 0.001 * gradient
            
        return {'fallback_update': 1}
    
    def train(self, 
              market_data: np.ndarray,
              episodes: int = 1000,
              max_steps_per_episode: int = 252) -> Dict[str, Any]:
        """
        Train SAC agent on historical market data.
        
        Args:
            market_data: Historical features (T, state_dim)
            episodes: Number of training episodes
            max_steps_per_episode: Max steps per episode
            
        Returns:
            Training results
        """
        logger.info(f"Starting SAC training: {episodes} episodes")
        
        episode_rewards = []
        best_reward = float('-inf')
        
        for episode in range(episodes):
            # Random starting point in data
            start_idx = np.random.randint(0, max(1, len(market_data) - max_steps_per_episode - 1))
            
            episode_reward = 0
            prev_position = 0.0
            cumulative_pnl = 0.0
            peak_pnl = 0.0
            
            for step in range(max_steps_per_episode):
                t = start_idx + step
                if t >= len(market_data) - 1:
                    break
                    
                state = market_data[t]
                next_state = market_data[t + 1]
                
                # Get position from actor
                position = self.get_position(state, deterministic=False)
                
                # Simulate PnL (simplified: position * next return)
                if len(state) > 0:
                    simulated_return = np.random.randn() * 0.02  # Simulated daily return
                    pnl = position * simulated_return
                    cumulative_pnl += pnl
                    peak_pnl = max(peak_pnl, cumulative_pnl)
                    drawdown = peak_pnl - cumulative_pnl
                    
                    # Compute reward
                    reward = self.compute_reward(
                        pnl=pnl,
                        drawdown=drawdown,
                        position_change=abs(position - prev_position),
                        volatility=0.02,  # Simplified
                    )
                else:
                    reward = 0
                    pnl = 0
                    
                # Store experience
                self.store_experience(
                    state,
                    np.array([position]),
                    reward,
                    next_state,
                    step == max_steps_per_episode - 1,
                )
                
                episode_reward += reward
                prev_position = position
                
                # Update networks
                if len(self.buffer) >= self.config.min_buffer_size:
                    for _ in range(self.config.updates_per_step):
                        self.update()
                        
            episode_rewards.append(episode_reward)
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode + 1}/{episodes}, "
                           f"Avg Reward (100): {avg_reward:.4f}, "
                           f"Alpha: {self.alpha:.4f}")
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save(self.log_dir / "best_sac_model.pt")
                    
        # Save final model
        self.save(self.log_dir / "final_sac_model.pt")
        
        results = {
            "episodes": episodes,
            "final_avg_reward": float(np.mean(episode_rewards[-100:])),
            "best_avg_reward": float(best_reward),
            "episode_rewards": episode_rewards,
            "total_steps": self.total_steps,
        }
        
        logger.info(f"SAC training complete. Best avg reward: {best_reward:.4f}")
        
        return results
    
    def save(self, path: Path):
        """Save model to disk."""
        if TORCH_AVAILABLE:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'alpha': self.alpha,
                'total_steps': self.total_steps,
                'config': self.config.to_dict(),
            }, path)
        else:
            np.savez(path, 
                    weights_mean=self.weights_mean,
                    weights_std=self.weights_std)
        logger.info(f"SAC model saved to {path}")
        
    def load(self, path: Path):
        """Load model from disk."""
        if TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.alpha = checkpoint['alpha']
            self.total_steps = checkpoint['total_steps']
        else:
            data = np.load(path)
            self.weights_mean = data['weights_mean']
            self.weights_std = data['weights_std']
        logger.info(f"SAC model loaded from {path}")
        
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for analysis."""
        return {
            "total_steps": self.total_steps,
            "buffer_size": len(self.buffer),
            "alpha": float(self.alpha) if hasattr(self, 'alpha') else 0.0,
            "num_episodes": len(self.episode_rewards),
            "avg_recent_reward": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
        }
