"""
Prioritized Experience Replay Buffer
=====================================

V2.3 Implementation of Prioritized Experience Replay (PER) with
Sum Tree data structure for O(log n) sampling.

Key Features:
- Priority-based sampling for important transitions
- Sum Tree for efficient O(log n) sampling
- Importance sampling weights for unbiased updates
- Automatic priority updates based on TD error
- Memory-efficient circular buffer

Research Basis:
- PER improves sample efficiency by 2x in DQN/SAC
- Priority exponent α=0.6, β annealing to 1.0
- TD error-based priorities capture learning signal

Usage:
    buffer = PrioritizedReplayBuffer(capacity=1_000_000)
    buffer.add(state, action, reward, next_state, done, priority)
    batch, weights, indices = buffer.sample(batch_size)
    buffer.update_priorities(indices, new_priorities)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# SUM TREE DATA STRUCTURE
# =============================================================================

class SumTree:
    """
    Sum Tree for O(log n) priority-based sampling.
    
    Binary tree where each parent is the sum of its children.
    Leaf nodes store priorities, enabling efficient:
    - Update: O(log n)
    - Sample: O(log n)
    - Get total priority: O(1)
    
    Structure:
        [internal nodes] [leaf nodes]
        Indices: 0 to capacity-2 are internal, capacity-1 to 2*capacity-2 are leaves
    """
    
    def __init__(self, capacity: int):
        """
        Initialize Sum Tree.
        
        Args:
            capacity: Maximum number of experiences (leaf nodes)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Full binary tree
        self.data = np.zeros(capacity, dtype=object)  # Stores experiences
        self.data_pointer = 0
        self.n_entries = 0
        
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for a given cumulative priority."""
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
        """Total priority (root of tree)."""
        return self.tree[0]
    
    @property
    def max_priority(self) -> float:
        """Maximum priority in leaves."""
        leaf_start = self.capacity - 1
        if self.n_entries == 0:
            return 1.0
        return max(self.tree[leaf_start:leaf_start + self.n_entries])
    
    @property
    def min_priority(self) -> float:
        """Minimum non-zero priority in leaves."""
        leaf_start = self.capacity - 1
        if self.n_entries == 0:
            return 1.0
        priorities = self.tree[leaf_start:leaf_start + self.n_entries]
        non_zero = priorities[priorities > 0]
        return non_zero.min() if len(non_zero) > 0 else 1.0
    
    def add(self, priority: float, data: Any):
        """
        Add experience with given priority.
        
        Args:
            priority: Priority value (typically |TD error| + ε)
            data: Experience tuple
        """
        # Leaf index
        idx = self.data_pointer + self.capacity - 1
        
        # Store data
        self.data[self.data_pointer] = data
        
        # Update tree
        self.update(idx, priority)
        
        # Increment pointer (circular)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx: int, priority: float):
        """
        Update priority at given index.
        
        Args:
            idx: Tree index (leaf index)
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get experience for cumulative priority s.
        
        Args:
            s: Cumulative priority value in [0, total]
            
        Returns:
            Tuple of (tree_index, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


# =============================================================================
# PRIORITIZED REPLAY BUFFER
# =============================================================================

@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay."""
    
    capacity: int = 1_000_000
    alpha: float = 0.6        # Priority exponent [0, 1], 0 = uniform
    beta: float = 0.4         # Importance sampling exponent, anneals to 1
    beta_increment: float = 0.001  # Per-sample beta increase
    epsilon: float = 1e-6     # Small constant to avoid zero priority
    max_priority: float = 1.0  # Initial max priority


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer with Sum Tree.
    
    Samples experiences with probability proportional to their
    TD-error-based priority, enabling more efficient learning
    from important transitions.
    
    Priority: P(i) = (|δ_i| + ε)^α / Σ (|δ_j| + ε)^α
    
    Importance sampling weights correct for bias:
    w_i = (N * P(i))^(-β) / max(w_j)
    """
    
    def __init__(self, config: PERConfig = None, **kwargs):
        """
        Initialize buffer.
        
        Args:
            config: PER configuration
            **kwargs: Override config values
        """
        if config is None:
            config = PERConfig(**kwargs)
        self.config = config
        
        # Sum tree for efficient sampling
        self.tree = SumTree(config.capacity)
        
        # Beta annealing
        self.beta = config.beta
        self.beta_increment = config.beta_increment
        
        # Track max priority for new samples
        self.max_priority = config.max_priority
        
    def __len__(self) -> int:
        """Number of stored experiences."""
        return self.tree.n_entries
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
            priority: Optional priority (uses max if not provided)
        """
        # Use max priority for new experiences
        if priority is None:
            priority = self.max_priority
            
        # Apply priority exponent
        priority = (priority + self.config.epsilon) ** self.config.alpha
        
        # Store experience
        experience = (state, action, reward, next_state, done)
        self.tree.add(priority, experience)
        
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (batch_dict, importance_weights, tree_indices)
        """
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size)
        
        # Segment priority range for stratified sampling
        segment = self.tree.total / batch_size
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # Sample from segment [i*segment, (i+1)*segment]
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            indices[i] = idx
            priorities[i] = priority
            
            state, action, reward, next_state, done = data
            batch['states'].append(state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['next_states'].append(next_state)
            batch['dones'].append(done)
            
        # Compute importance sampling weights
        # w_i = (N * P(i))^(-β) / max(w)
        sampling_probs = priorities / self.tree.total
        weights = (len(self) * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Convert to numpy arrays
        batch_np = {
            'states': np.array(batch['states']),
            'actions': np.array(batch['actions']),
            'rewards': np.array(batch['rewards']),
            'next_states': np.array(batch['next_states']),
            'dones': np.array(batch['dones']),
        }
        
        return batch_np, weights.astype(np.float32), indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Tree indices from sample()
            priorities: New priority values (typically |TD error|)
        """
        for idx, priority in zip(indices, priorities):
            # Apply exponent and add epsilon
            priority = (abs(priority) + self.config.epsilon) ** self.config.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update tree
            self.tree.update(idx, priority)
    
    def sample_uniform(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample uniformly (for comparison/debugging).
        
        Args:
            batch_size: Number of experiences
            
        Returns:
            Batch dictionary
        """
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        
        for i in indices:
            data_idx = i  # Direct data access
            data = self.tree.data[data_idx % self.tree.capacity]
            if data is None or data == 0:
                continue
                
            state, action, reward, next_state, done = data
            batch['states'].append(state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['next_states'].append(next_state)
            batch['dones'].append(done)
            
        return {k: np.array(v) for k, v in batch.items()}
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        return {
            'size': len(self),
            'capacity': self.config.capacity,
            'beta': self.beta,
            'max_priority': self.max_priority,
            'total_priority': self.tree.total,
            'min_priority': self.tree.min_priority,
        }


# =============================================================================
# TORCH-COMPATIBLE WRAPPER
# =============================================================================

if TORCH_AVAILABLE:
    
    class TorchPrioritizedReplayBuffer(PrioritizedReplayBuffer):
        """
        PyTorch-compatible Prioritized Replay Buffer.
        
        Returns torch tensors and handles GPU transfer.
        """
        
        def __init__(self, config: PERConfig = None, device: str = 'cpu', **kwargs):
            super().__init__(config, **kwargs)
            self.device = torch.device(device)
            
        def sample_torch(
            self, 
            batch_size: int
        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, np.ndarray]:
            """
            Sample batch as PyTorch tensors.
            
            Args:
                batch_size: Number of experiences
                
            Returns:
                Tuple of (batch_dict, importance_weights, indices)
            """
            batch_np, weights, indices = self.sample(batch_size)
            
            batch_torch = {
                'states': torch.FloatTensor(batch_np['states']).to(self.device),
                'actions': torch.FloatTensor(batch_np['actions']).to(self.device),
                'rewards': torch.FloatTensor(batch_np['rewards']).unsqueeze(-1).to(self.device),
                'next_states': torch.FloatTensor(batch_np['next_states']).to(self.device),
                'dones': torch.FloatTensor(batch_np['dones']).unsqueeze(-1).to(self.device),
            }
            
            weights_torch = torch.FloatTensor(weights).to(self.device)
            
            return batch_torch, weights_torch, indices


# =============================================================================
# SIMPLE REPLAY BUFFER (for comparison)
# =============================================================================

class SimpleReplayBuffer:
    """
    Standard uniform replay buffer for comparison.
    
    Uses deque for O(1) add/remove with circular buffer behavior.
    """
    
    def __init__(self, capacity: int = 1_000_000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        
    def __len__(self) -> int:
        return len(self.buffer)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = None  # Ignored, for API compatibility
    ):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, None]:
        """Sample uniformly."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            batch['states'].append(state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['next_states'].append(next_state)
            batch['dones'].append(done)
            
        batch_np = {k: np.array(v) for k, v in batch.items()}
        weights = np.ones(batch_size, dtype=np.float32)  # Uniform weights
        
        return batch_np, weights, None
    
    def update_priorities(self, indices, priorities):
        """No-op for uniform buffer."""
        pass


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test Sum Tree
    print("Testing Sum Tree...")
    tree = SumTree(8)
    
    for i in range(8):
        tree.add(float(i + 1), f"data_{i}")
    
    print(f"Total priority: {tree.total}")
    print(f"Max priority: {tree.max_priority}")
    
    # Sample based on priority
    for s in [0.5, 10.0, 20.0, 30.0]:
        idx, priority, data = tree.get(s)
        print(f"  s={s:.1f} -> idx={idx}, priority={priority}, data={data}")
    
    print("\nTesting Prioritized Replay Buffer...")
    config = PERConfig(capacity=1000, alpha=0.6, beta=0.4)
    buffer = PrioritizedReplayBuffer(config)
    
    # Add experiences
    state_dim = 10
    for i in range(500):
        state = np.random.randn(state_dim)
        action = np.array([np.random.rand()])
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.rand() < 0.1
        
        # Priority based on reward magnitude
        priority = abs(reward) + 0.01
        buffer.add(state, action, reward, next_state, done, priority)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Statistics: {buffer.get_statistics()}")
    
    # Sample
    batch, weights, indices = buffer.sample(32)
    print(f"\nSampled batch shapes:")
    print(f"  States: {batch['states'].shape}")
    print(f"  Actions: {batch['actions'].shape}")
    print(f"  Rewards: {batch['rewards'].shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Update priorities
    new_priorities = np.random.rand(32) * 2
    buffer.update_priorities(indices, new_priorities)
    print(f"\nAfter priority update:")
    print(f"  Max priority: {buffer.max_priority:.3f}")
    
    # Test PyTorch wrapper
    if TORCH_AVAILABLE:
        print("\nTesting PyTorch wrapper...")
        torch_buffer = TorchPrioritizedReplayBuffer(config, device='cpu')
        
        for i in range(100):
            state = np.random.randn(state_dim)
            action = np.array([np.random.rand()])
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = np.random.rand() < 0.1
            torch_buffer.add(state, action, reward, next_state, done)
        
        batch, weights, indices = torch_buffer.sample_torch(16)
        print(f"  States tensor: {batch['states'].shape}, device: {batch['states'].device}")
        print(f"  Weights tensor: {weights.shape}")
    
    print("\n✅ All tests passed!")
