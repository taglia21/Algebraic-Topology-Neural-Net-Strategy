"""
Machine Learning Module for V2.0 Trading System

This module contains advanced ML components:
- AdaptiveEnsemble: Self-training production ensemble (primary)
- TransformerPredictor: Attention-based stock direction prediction
- StackedEnsemble: Stacked ensemble learner
- SACAgent: Soft Actor-Critic with Prioritized Experience Replay
- OnlineLearner: SGD-based online learning with signal confidence
- build_features: Feature engineering for ML pipeline
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

try:
    from .online_learner import OnlineLearner, OnlineLearnerConfig, TradeOutcome
except ImportError:
    pass

try:
    from .feature_engineering import build_features
except ImportError:
    build_features = None

__all__ = [
    'AdaptiveEnsemble',
    'TransformerPredictor',
    'StackedEnsemble',
    'SACAgent',
    'SACConfig',
    'PrioritizedReplayBuffer',
    'Experience',
    'OnlineLearner',
    'OnlineLearnerConfig',
    'TradeOutcome',
    'build_features',
]
