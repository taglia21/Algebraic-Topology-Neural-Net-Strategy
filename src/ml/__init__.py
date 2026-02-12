"""
Machine Learning Module for V2.0 Trading System

This module contains advanced ML components:
- AdaptiveEnsemble: Self-training production ensemble (primary)
- TransformerPredictor: Attention-based stock direction prediction
- StackedEnsemble: Stacked ensemble learner
- SACAgent: Soft Actor-Critic with Prioritized Experience Replay
"""

try:
    from .adaptive_ensemble import AdaptiveEnsemble
except ImportError as e:
    AdaptiveEnsemble = None

try:
    from .transformer_predictor import TransformerPredictor
except ImportError as e:
    TransformerPredictor = None

try:
    from .stacked_ensemble import StackedEnsemble
except ImportError as e:
    StackedEnsemble = None

try:
    from .sac_agent import SACAgent, SACConfig, PrioritizedReplayBuffer, Experience
except ImportError as e:
    SACAgent = None
    SACConfig = None
    PrioritizedReplayBuffer = None
    Experience = None

__all__ = [
    'AdaptiveEnsemble',
    'TransformerPredictor',
    'StackedEnsemble',
    'SACAgent',
    'SACConfig',
    'PrioritizedReplayBuffer',
    'Experience'
]
