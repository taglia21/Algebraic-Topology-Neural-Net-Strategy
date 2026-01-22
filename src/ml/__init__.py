"""
Machine Learning Module for V2.0 Trading System

This module contains advanced ML components:
- TransformerPredictor: Attention-based stock direction prediction
- SACAgent: Soft Actor-Critic with Prioritized Experience Replay
- PrioritizedReplayBuffer: Experience replay with priority sampling
- SACConfig: Configuration for SAC agent
"""

try:
    from .transformer_predictor import TransformerPredictor
except ImportError as e:
    TransformerPredictor = None

try:
    from .sac_agent import SACAgent, SACConfig, PrioritizedReplayBuffer, Experience
except ImportError as e:
    SACAgent = None
    SACConfig = None
    PrioritizedReplayBuffer = None
    Experience = None

__all__ = [
    'TransformerPredictor',
    'SACAgent',
    'SACConfig',
    'PrioritizedReplayBuffer',
    'Experience'
]
