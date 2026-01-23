"""
Soft Actor-Critic (SAC) Agent with Prioritized Experience Replay (PER)

Replaces Q-learning with continuous, entropy-regularized policy learning.

Key Features:
- Twin Q-networks for stable training
- Entropy regularization for exploration
- Automatic temperature tuning
- Prioritized Experience Replay (PER) for efficient learning
- Continuous action space: position sizing in [0.0, 2.0]

Target: 20% improvement in risk-adjusted returns, 40% faster convergence
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY (PER)
# =============================================================================

class SumTree:
    """
    Sum tree for O(log n) priority sampling.
    
    Based on: Schaul et al., "Prioritized Experience Replay" (2016)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index by priority value."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    @property
    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]
    
    @property
    def max_priority(self) -> float:
        """Maximum priority in buffer."""
        return max(self.tree[self.capacity-1:self.capacity-1+self.n_entries])
    
    def add(self, priority: float, data):
        """Add experience with priority."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, any]:
        """Get experience by priority value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class Experience(NamedTuple):
    """Single experience tuple."""
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using Sum Tree.
    
    Priority calculation: P(i) = p_i^α / Σ p_j^α
    Priority value: p_i = |TD_error| + ε
    
    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent (0=uniform, 1=full prioritization)
        beta_start: Initial importance sampling weight
        beta_end: Final importance sampling weight
        beta_frames: Frames to anneal beta
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_frames: int = 100000, epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0
    
    @property
    def beta(self) -> float:
        """Current importance sampling weight."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def add(self, experience: Experience, td_error: Optional[float] = None):
        """
        Add experience to buffer.
        
        Args:
            experience: Experience tuple
            td_error: TD error for priority (uses max if None)
        """
        priority = (abs(td_error) + self.epsilon) ** self.alpha if td_error else self.max_priority
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """
        Sample batch with priorities.
        
        Returns:
            experiences: List of experience tuples
            weights: Importance sampling weights
            indices: Tree indices for priority update
        """
        experiences = []
        indices = []
        priorities = []
        
        segment = self.tree.total / batch_size
        
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, exp = self.tree.get(s)
            
            if exp is not None:
                experiences.append(exp)
                indices.append(idx)
                priorities.append(priority)
        
        # Importance sampling weights
        probs = np.array(priorities) / self.tree.total
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        self.frame += batch_size
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries


# =============================================================================
# SAC NETWORKS (Only defined when PyTorch is available)
# =============================================================================

if TORCH_AVAILABLE:
    class MLP(nn.Module):
        """Simple MLP with configurable layers."""
        
        def __init__(self, input_dim: int, output_dim: int, 
                     hidden_dims: List[int] = [256, 256],
                     activation=None):
            super().__init__()
            if activation is None:
                activation = nn.ReLU
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activation())
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x)


    class GaussianActor(nn.Module):
        """
        Gaussian policy network for continuous actions.
        
        Outputs mean and log_std of Gaussian distribution.
        Actions are bounded using tanh squashing.
        """
        
        LOG_STD_MIN = -20
        LOG_STD_MAX = 2
        
        def __init__(self, state_dim: int, action_dim: int = 1,
                     hidden_dims: List[int] = [256, 256],
                     action_scale: float = 1.0, action_bias: float = 1.0):
            """
            Args:
                state_dim: State dimension
                action_dim: Action dimension
                hidden_dims: Hidden layer dimensions
                action_scale: Scale for tanh output
                action_bias: Bias for tanh output (action = scale * tanh + bias)
            """
            super().__init__()
            
            self.action_scale = action_scale
            self.action_bias = action_bias
            
            # Shared backbone
            self.backbone = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1])
            
            # Mean and log_std heads
            self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        def forward(self, state):
            """Output mean and log_std."""
            features = F.relu(self.backbone(state))
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            return mean, log_std
        
        def sample(self, state):
            """
            Sample action and compute log probability.
            
            Returns:
                action: Sampled action (scaled)
                log_prob: Log probability of action
            """
            mean, log_std = self(state)
            std = log_std.exp()
            
            # Sample from Gaussian
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            
            # Squash through tanh
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            
            # Log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
            
            return action, log_prob
        
        def deterministic_action(self, state):
            """Get deterministic action (mean)."""
            mean, _ = self(state)
            return torch.tanh(mean) * self.action_scale + self.action_bias


    class TwinQNetwork(nn.Module):
        """Twin Q-networks for SAC (reduces overestimation bias)."""
        
        def __init__(self, state_dim: int, action_dim: int = 1,
                     hidden_dims: List[int] = [256, 256]):
            super().__init__()
            
            self.q1 = MLP(state_dim + action_dim, 1, hidden_dims)
            self.q2 = MLP(state_dim + action_dim, 1, hidden_dims)
        
        def forward(self, state, action):
            """Compute Q-values from both networks."""
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)
        
        def q1_forward(self, state, action):
            """Compute Q1 only (for policy update)."""
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa)
else:
    # Dummy classes when PyTorch not available
    MLP = None
    GaussianActor = None
    TwinQNetwork = None


# =============================================================================
# SAC AGENT
# =============================================================================

@dataclass
class SACConfig:
    """SAC hyperparameters."""
    state_dim: int = 27  # Base features + regime + TDA
    action_dim: int = 1  # Position sizing multiplier
    hidden_dims: List[int] = None
    gamma: float = 0.99
    tau: float = 0.005
    alpha_init: float = 0.2
    auto_alpha: bool = True
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    action_scale: float = 1.0
    action_bias: float = 1.0  # Action range: [0, 2]
    buffer_size: int = 100000
    batch_size: int = 256
    warmup_steps: int = 1000
    update_every: int = 1
    
    # V26 Enhancements
    use_curiosity: bool = True              # Intrinsic curiosity rewards
    curiosity_scale: float = 0.1            # Curiosity reward scaling
    curiosity_lr: float = 1e-4              # Curiosity module learning rate
    
    use_multi_timeframe: bool = True        # Multi-timeframe actions
    hourly_weight: float = 0.5              # Weight for 1hr entry signals
    daily_weight: float = 0.3               # Weight for daily sizing
    weekly_weight: float = 0.2              # Weight for weekly rotation
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


# =============================================================================
# V26 INTRINSIC CURIOSITY MODULE (ICM)
# =============================================================================

if TORCH_AVAILABLE:
    class IntrinsicCuriosityModule(nn.Module):
        """
        Intrinsic Curiosity Module for exploration.
        
        Provides intrinsic reward based on prediction error of next state.
        Encourages exploration of novel states.
        
        Based on: Pathak et al., "Curiosity-driven Exploration" (2017)
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            
            # State encoder (inverse model)
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
            
            # Forward model: predict next encoded state from current + action
            self.forward_model = nn.Sequential(
                nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
            
            # Inverse model: predict action from state pair
            self.inverse_model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, state, action, next_state):
            """
            Compute curiosity reward.
            
            Returns:
                curiosity_reward: Intrinsic reward based on prediction error
                forward_loss: Forward model loss for training
                inverse_loss: Inverse model loss for training
            """
            # Encode states
            phi_state = self.state_encoder(state)
            phi_next_state = self.state_encoder(next_state)
            
            # Forward model prediction
            forward_input = torch.cat([phi_state, action], dim=-1)
            pred_phi_next = self.forward_model(forward_input)
            
            # Forward loss (prediction error = curiosity)
            forward_loss = 0.5 * (pred_phi_next - phi_next_state.detach()).pow(2).sum(-1)
            
            # Inverse model prediction
            inverse_input = torch.cat([phi_state, phi_next_state], dim=-1)
            pred_action = self.inverse_model(inverse_input)
            
            # Inverse loss
            inverse_loss = 0.5 * (pred_action - action).pow(2).sum(-1)
            
            # Curiosity reward is forward loss (novel = high error = high reward)
            curiosity_reward = forward_loss.detach()
            
            return curiosity_reward, forward_loss.mean(), inverse_loss.mean()
else:
    IntrinsicCuriosityModule = None


# =============================================================================
# V26 MULTI-TIMEFRAME CONTROLLER
# =============================================================================

class MultiTimeframeController:
    """
    Multi-timeframe action controller.
    
    Combines signals from multiple timeframes:
    - 1hr: Entry timing
    - Daily: Position sizing
    - Weekly: Sector rotation
    """
    
    def __init__(self, config: 'SACConfig'):
        self.config = config
        
        # Historical signals by timeframe
        self.hourly_signals: List[float] = []
        self.daily_signals: List[float] = []
        self.weekly_signals: List[float] = []
        
        # Current combined signal
        self.current_signal: float = 0.0
    
    def update_hourly(self, signal: float):
        """Update 1hr entry signal."""
        self.hourly_signals.append(signal)
        if len(self.hourly_signals) > 24:  # Keep last 24 hours
            self.hourly_signals = self.hourly_signals[-24:]
    
    def update_daily(self, signal: float):
        """Update daily sizing signal."""
        self.daily_signals.append(signal)
        if len(self.daily_signals) > 20:  # Keep last 20 days
            self.daily_signals = self.daily_signals[-20:]
    
    def update_weekly(self, signal: float):
        """Update weekly rotation signal."""
        self.weekly_signals.append(signal)
        if len(self.weekly_signals) > 12:  # Keep last 12 weeks
            self.weekly_signals = self.weekly_signals[-12:]
    
    def get_combined_signal(self) -> float:
        """
        Get weighted combination of signals.
        
        Returns:
            Combined signal in [0, 1]
        """
        signals = []
        weights = []
        
        if self.hourly_signals:
            # Use most recent hourly + short-term momentum
            hourly_val = np.mean(self.hourly_signals[-4:]) if len(self.hourly_signals) >= 4 else self.hourly_signals[-1]
            signals.append(hourly_val)
            weights.append(self.config.hourly_weight)
        
        if self.daily_signals:
            # Use daily trend
            daily_val = np.mean(self.daily_signals[-5:]) if len(self.daily_signals) >= 5 else self.daily_signals[-1]
            signals.append(daily_val)
            weights.append(self.config.daily_weight)
        
        if self.weekly_signals:
            # Use weekly regime
            weekly_val = np.mean(self.weekly_signals[-2:]) if len(self.weekly_signals) >= 2 else self.weekly_signals[-1]
            signals.append(weekly_val)
            weights.append(self.config.weekly_weight)
        
        if not signals:
            return 0.5  # Neutral
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        combined = sum(s * w for s, w in zip(signals, weights))
        self.current_signal = float(np.clip(combined, 0, 1))
        
        return self.current_signal
    
    def get_entry_timing_multiplier(self) -> float:
        """Get entry timing multiplier from hourly signals."""
        if len(self.hourly_signals) < 4:
            return 1.0
        
        # Check for momentum alignment
        recent = self.hourly_signals[-4:]
        trend = (recent[-1] - recent[0]) / 4  # Slope
        
        if trend > 0.05:  # Strong uptrend
            return 1.2
        elif trend < -0.05:  # Strong downtrend
            return 0.8
        return 1.0
    
    def get_sizing_multiplier(self) -> float:
        """Get position sizing multiplier from daily signals."""
        if len(self.daily_signals) < 5:
            return 1.0
        
        # Higher sizing when daily trend is strong
        daily_std = np.std(self.daily_signals[-20:]) if len(self.daily_signals) >= 20 else 0.1
        daily_mean = np.mean(self.daily_signals[-5:])
        
        if daily_mean > 0.6 and daily_std < 0.15:  # Strong, stable trend
            return 1.3
        elif daily_mean < 0.4 or daily_std > 0.25:  # Weak or volatile
            return 0.7
        return 1.0
    
    def get_rotation_signal(self) -> str:
        """Get weekly rotation signal."""
        if len(self.weekly_signals) < 2:
            return "hold"
        
        recent_avg = np.mean(self.weekly_signals[-2:])
        
        if recent_avg > 0.65:
            return "aggressive"
        elif recent_avg < 0.35:
            return "defensive"
        return "neutral"


class SACAgent:
    """
    Soft Actor-Critic agent with Prioritized Experience Replay.
    
    For position sizing: continuous action in [0, 2] representing
    position size multiplier.
    
    State features:
    - Stock features (momentum, volatility, etc.)
    - Regime indicators
    - TDA features
    - Current position info
    """
    
    def __init__(self, config: Optional[SACConfig] = None,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize SAC agent.
        
        Args:
            config: SAC configuration
            model_path: Path to save/load model
            device: Compute device
        """
        self.config = config or SACConfig()
        self.model_path = model_path or "models/sac_agent.pt"
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - SAC agent disabled")
            self.actor = None
            return
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"SAC using device: {self.device}")
        
        # Networks
        self.actor = GaussianActor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            self.config.action_scale,
            self.config.action_bias
        ).to(self.device)
        
        self.critic = TwinQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.device)
        
        self.critic_target = TwinQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.config.alpha_init
        self.target_entropy = -self.config.action_dim
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
        
        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_frames=100000
        )
        
        # V26: Intrinsic Curiosity Module
        self.curiosity_module = None
        self.curiosity_optimizer = None
        if self.config.use_curiosity and IntrinsicCuriosityModule is not None:
            self.curiosity_module = IntrinsicCuriosityModule(
                self.config.state_dim,
                self.config.action_dim
            ).to(self.device)
            self.curiosity_optimizer = torch.optim.Adam(
                self.curiosity_module.parameters(),
                lr=self.config.curiosity_lr
            )
            logger.info("V26: Intrinsic Curiosity Module enabled")
        
        # V26: Multi-timeframe controller
        self.mtf_controller = None
        if self.config.use_multi_timeframe:
            self.mtf_controller = MultiTimeframeController(self.config)
            logger.info("V26: Multi-timeframe controller enabled")
        
        # Training state
        self.total_steps = 0
        self.updates = 0
        self.curiosity_rewards_total = 0.0
        
        # Try to load saved model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load model if exists."""
        try:
            import os
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])
                self.critic_target.load_state_dict(checkpoint['critic_target'])
                self.log_alpha = checkpoint.get('log_alpha', self.log_alpha)
                self.total_steps = checkpoint.get('total_steps', 0)
                logger.info(f"Loaded SAC checkpoint from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not load SAC model: {e}")
        return False
    
    def save_model(self):
        """Save model checkpoint."""
        if self.actor is None:
            return
        
        import os
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'total_steps': self.total_steps,
        }
        torch.save(checkpoint, self.model_path)
        logger.info(f"Saved SAC checkpoint to {self.model_path}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """
        Select action given state.
        
        Args:
            state: State observation
            deterministic: If True, use mean action (no exploration)
        
        Returns:
            action: Position sizing multiplier [0, 2]
        """
        if self.actor is None:
            return 1.0  # Default multiplier
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                action = self.actor.deterministic_action(state_t)
            else:
                action, _ = self.actor.sample(state_t)
            
            return action.cpu().numpy()[0, 0]
    
    def add_experience(self, state: np.ndarray, action: float, reward: float,
                       next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """Add experience to replay buffer."""
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.add(exp, td_error)
        self.total_steps += 1
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one update step.
        
        Returns:
            Dictionary of training metrics or None if not updating
        """
        if self.actor is None:
            return None
        
        if len(self.buffer) < self.config.warmup_steps:
            return None
        
        if self.total_steps % self.config.update_every != 0:
            return None
        
        # Sample from buffer
        experiences, weights, indices = self.buffer.sample(self.config.batch_size)
        
        if len(experiences) < self.config.batch_size:
            return None
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.FloatTensor(np.array([[e.action] for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([[e.reward] for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([[e.done] for e in experiences])).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # V26: Add intrinsic curiosity rewards
        curiosity_loss = 0.0
        if self.curiosity_module is not None:
            curiosity_reward, forward_loss, inverse_loss = self.curiosity_module(
                states, actions, next_states
            )
            # Scale and add to extrinsic rewards
            intrinsic_rewards = curiosity_reward.unsqueeze(1) * self.config.curiosity_scale
            rewards = rewards + intrinsic_rewards
            self.curiosity_rewards_total += intrinsic_rewards.mean().item()
            
            # Update curiosity module
            curiosity_loss = forward_loss + 0.2 * inverse_loss
            self.curiosity_optimizer.zero_grad()
            curiosity_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.curiosity_module.parameters(), 1.0)
            self.curiosity_optimizer.step()
            curiosity_loss = curiosity_loss.item()
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.config.gamma * (1 - dones) * q_next
        
        q1, q2 = self.critic(states, actions)
        
        # TD errors for PER
        td_errors = (q_target - torch.min(q1, q2)).abs().detach().cpu().numpy().flatten()
        self.buffer.update_priorities(indices, td_errors)
        
        # Weighted critic loss
        critic_loss = (weights_t * ((q1 - q_target).pow(2) + (q2 - q_target).pow(2))).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic.q1_forward(states, new_actions)
        actor_loss = (self.alpha * log_probs - q1_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        self.updates += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'mean_q': q1.mean().item(),
            'mean_td_error': td_errors.mean(),
            'curiosity_loss': curiosity_loss if isinstance(curiosity_loss, float) else 0.0
        }
    
    def compute_position_multiplier(self, state: np.ndarray, 
                                    vol_20d: float = 0.02, 
                                    vix: float = 20.0,
                                    deterministic: bool = True,
                                    hourly_signal: Optional[float] = None) -> float:
        """
        Compute position sizing multiplier with dynamic scaling.
        
        V26 Enhanced: Includes multi-timeframe adjustments.
        
        Args:
            state: Current state features
            vol_20d: 20-day realized volatility
            vix: Current VIX level
            deterministic: Use deterministic action
            hourly_signal: Optional 1hr signal for MTF (V26)
        
        Returns:
            Position multiplier (typically 0.25-2.0)
        """
        # Base action from SAC
        base_action = self.select_action(state, deterministic)
        
        # Dynamic bounds based on volatility
        vol_scale = 1.0 / np.sqrt(vol_20d + 1e-6)
        vol_scale = np.clip(vol_scale, 0.5, 2.0)
        
        # VIX regime adjustment
        if vix > 25:  # High fear
            regime_scale = 0.75
        elif vix < 15:  # Complacency
            regime_scale = 1.5
        else:  # Normal
            regime_scale = 1.0
        
        # V26: Multi-timeframe adjustment
        mtf_scale = 1.0
        if self.mtf_controller is not None:
            if hourly_signal is not None:
                self.mtf_controller.update_hourly(hourly_signal)
            mtf_scale = self.mtf_controller.get_entry_timing_multiplier() * \
                       self.mtf_controller.get_sizing_multiplier()
        
        # Final multiplier
        multiplier = base_action * vol_scale * regime_scale * mtf_scale
        
        # Clamp to safe range
        return float(np.clip(multiplier, 0.25, 2.0))
    
    def update_daily_signal(self, signal: float):
        """V26: Update daily signal for multi-timeframe controller."""
        if self.mtf_controller is not None:
            self.mtf_controller.update_daily(signal)
    
    def update_weekly_signal(self, signal: float):
        """V26: Update weekly signal for multi-timeframe controller."""
        if self.mtf_controller is not None:
            self.mtf_controller.update_weekly(signal)
    
    def get_rotation_recommendation(self) -> str:
        """V26: Get weekly rotation recommendation."""
        if self.mtf_controller is not None:
            return self.mtf_controller.get_rotation_signal()
        return "neutral"
    
    def get_stats(self) -> Dict[str, any]:
        """Get current agent statistics."""
        stats = {
            'total_steps': self.total_steps,
            'updates': self.updates,
            'buffer_size': len(self.buffer),
            'alpha': self.alpha,
            'beta': self.buffer.beta,
        }
        
        # V26 stats
        if self.curiosity_module is not None:
            stats['curiosity_rewards_total'] = self.curiosity_rewards_total
        
        if self.mtf_controller is not None:
            stats['mtf_combined_signal'] = self.mtf_controller.current_signal
            stats['rotation_signal'] = self.mtf_controller.get_rotation_signal()
        
        return stats


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class TradingEnvironment:
    """
    Simple trading environment for offline RL training.
    
    State: [stock_features..., regime_one_hot..., position_info...]
    Action: Position sizing multiplier [0, 2]
    Reward: Risk-adjusted return
    """
    
    def __init__(self, price_data: Dict[str, np.ndarray],
                 initial_capital: float = 100000):
        """
        Args:
            price_data: Dictionary of ticker -> price array
            initial_capital: Starting capital
        """
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.tickers = list(price_data.keys())
        
        self.current_ticker = None
        self.current_idx = 0
        self.position = 0.0
        self.capital = initial_capital
    
    def reset(self, ticker: Optional[str] = None) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_ticker = ticker or random.choice(self.tickers)
        self.current_idx = 50  # Start after warmup period
        self.position = 0.0
        self.capital = self.initial_capital
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        # Placeholder: would compute actual features
        # Returns normalized state vector
        return np.zeros(27)  # Match state_dim
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done.
        
        Args:
            action: Position sizing multiplier
        """
        prices = self.price_data[self.current_ticker]
        
        if self.current_idx >= len(prices) - 1:
            return self._get_state(), 0.0, True, {}
        
        # Current and next prices
        price = prices[self.current_idx]
        next_price = prices[self.current_idx + 1]
        
        # Return
        ret = (next_price - price) / price
        
        # Reward: position * return - transaction cost
        reward = self.position * action * ret - 0.0001 * abs(action - self.position)
        
        # Update position
        self.position = action
        self.current_idx += 1
        
        done = self.current_idx >= len(prices) - 1
        
        return self._get_state(), reward, done, {'return': ret}


def train_sac_offline(agent: SACAgent, 
                      experiences: List[Experience],
                      epochs: int = 10,
                      updates_per_epoch: int = 100) -> Dict[str, List[float]]:
    """
    Train SAC agent on offline experience data.
    
    Args:
        agent: SAC agent to train
        experiences: List of experience tuples
        epochs: Number of training epochs
        updates_per_epoch: Updates per epoch
    
    Returns:
        Training history dictionary
    """
    logger.info(f"Training SAC on {len(experiences)} experiences for {epochs} epochs")
    
    # Add all experiences to buffer
    for exp in experiences:
        agent.add_experience(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
    
    history = {'critic_loss': [], 'actor_loss': [], 'alpha': []}
    
    for epoch in range(epochs):
        epoch_metrics = {'critic_loss': [], 'actor_loss': [], 'alpha': []}
        
        for _ in range(updates_per_epoch):
            agent.total_steps += 1  # Force update
            metrics = agent.update()
            
            if metrics:
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)
        
        # Average metrics
        for k in history:
            if epoch_metrics[k]:
                history[k].append(np.mean(epoch_metrics[k]))
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Critic: {history['critic_loss'][-1]:.4f}, "
                       f"Actor: {history['actor_loss'][-1]:.4f}, "
                       f"Alpha: {history['alpha'][-1]:.4f}")
    
    agent.save_model()
    logger.info("SAC training complete")
    
    return history
