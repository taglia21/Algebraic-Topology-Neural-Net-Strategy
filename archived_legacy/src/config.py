"""Production Configuration for TDA+NN Trading System.

This file consolidates all configurable parameters for the production system.
Organized by component:
- Risk Management
- Transaction Costs  
- Data/Validation
- LSTM Model
- TDA Features
- Strategy Signals
- Walk-Forward Analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Position Sizing - OPTIMIZED: Half-Kelly for 60%+ win rate strategies
    kelly_fraction: float = 0.50  # Half-Kelly (industry standard for strong strategies)
    risk_per_trade: float = 0.02  # Risk 2% of equity per trade (justified by 60%+ win rate)
    max_position_pct: float = 0.15  # Max 15% of portfolio in single position
    
    # Stop Loss
    stop_atr_multiplier: float = 2.0  # ATR multiplier for stops
    min_stop_pct: float = 0.015  # Minimum 1.5% stop
    max_stop_pct: float = 0.04  # Maximum 4% stop
    atr_period: int = 14  # ATR calculation period
    
    # Take Profit
    risk_reward_ratio: float = 2.0  # 2:1 reward-to-risk ratio
    use_trailing_stop: bool = False  # Enable trailing stops
    trail_activation_pct: float = 0.02  # Activate after 2% profit
    trail_distance_pct: float = 0.01  # Trail by 1%
    
    # Portfolio Heat - OPTIMIZED: Allow 2-3 concurrent positions
    max_portfolio_heat: float = 0.35  # Max 35% total risk exposure
    max_correlated_positions: int = 3  # Limit correlated assets
    
    # Kelly Updates
    kelly_window: int = 50  # Trades for Kelly estimation
    min_kelly_trades: int = 20  # Minimum trades before Kelly


@dataclass
class CostConfig:
    """Transaction cost configuration."""
    
    # Commission Structure
    commission_per_trade: float = 1.0  # Base commission
    commission_per_share: float = 0.005  # Per-share commission
    
    # Market Impact
    spread_bps: float = 5.0  # Bid-ask spread in basis points
    slippage_bps: float = 3.0  # Slippage in basis points
    
    # Cost Scenario
    scenario: str = 'baseline'  # 'low_cost', 'baseline', 'high_cost', 'extreme'


@dataclass
class LSTMConfig:
    """LSTM model configuration."""
    
    # Architecture
    sequence_length: int = 30  # Input sequence length
    lstm_units: int = 64  # LSTM layer size
    n_dense_layers: int = 2  # Dense layers after LSTM
    dense_units: int = 32  # Dense layer size
    dropout_rate: float = 0.2  # Dropout for regularization
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    
    # Entropy Penalty (prevents overconfident predictions)
    use_entropy_penalty: bool = True
    entropy_weight: float = 0.05
    
    # Model checkpoint
    save_best_only: bool = True
    checkpoint_dir: str = 'results/'


@dataclass
class TDAConfig:
    """TDA (Topological Data Analysis) configuration."""
    
    # Sliding window for point cloud
    window: int = 20  # Window size for delay embedding
    embedding_dim: int = 3  # Takens embedding dimension
    
    # Persistence features
    persistence_threshold: float = 0.05  # Filter noise
    max_dimension: int = 1  # Compute H0, H1
    n_bins: int = 10  # Bins for persistence statistics
    
    # Feature mode
    feature_mode: str = 'v1.3'  # Feature version


@dataclass
class StrategyConfig:
    """Trading strategy configuration."""
    
    # Signal Thresholds
    buy_threshold: float = 0.55  # Probability to go long
    sell_threshold: float = 0.45  # Probability to exit/go flat
    min_confidence: float = 0.10  # Minimum prediction confidence
    
    # Ensemble Weights
    tda_weight: float = 0.3  # Weight for TDA signal
    lstm_weight: float = 0.7  # Weight for LSTM signal
    
    # Signal Smoothing
    smoothing_window: int = 3  # EMA window for signal smoothing
    min_holding_periods: int = 1  # Minimum bars to hold position
    
    # Position Management
    scale_in_enabled: bool = False  # Allow scaling into positions
    max_scale_in_levels: int = 3  # Maximum add-on levels


@dataclass
class ValidationConfig:
    """Data validation configuration."""
    
    # Date Range - 5 years for comprehensive backtesting
    start_date: str = '2020-01-01'  # Backtest start (5-year window)
    end_date: str = '2025-12-31'  # Backtest end
    
    # Data Provider - Polygon for high-quality data
    provider: str = 'polygon'  # 'yfinance', 'polygon', 'alpaca'
    
    # Quality Checks
    max_gap_days: int = 5  # Max allowable data gap
    min_price_pct: float = 0.001  # Minimum price change to detect stale data
    validate_ohlcv: bool = True  # Check OHLCV integrity
    
    # Walk-Forward
    train_months: int = 24  # Training window
    test_months: int = 6  # Testing window
    step_months: int = 3  # Step size between windows
    min_windows: int = 8  # Minimum windows required


@dataclass
class RegimeConfig:
    """Market regime detection configuration."""
    
    # Moving Averages
    ma_fast: int = 50  # Fast MA period
    ma_slow: int = 200  # Slow MA period
    
    # Momentum/Volatility
    rsi_period: int = 14  # RSI period
    atr_period: int = 14  # ATR period
    
    # Regime Thresholds
    bull_rsi_min: float = 45  # RSI floor for bull market
    bear_rsi_max: float = 55  # RSI ceiling for bear market
    
    # Volatility States
    high_vol_percentile: float = 75  # High volatility threshold
    low_vol_percentile: float = 25  # Low volatility threshold


class ProductionConfig:
    """Master configuration class combining all components."""
    
    def __init__(
        self,
        risk: RiskConfig = None,
        costs: CostConfig = None,
        lstm: LSTMConfig = None,
        tda: TDAConfig = None,
        strategy: StrategyConfig = None,
        validation: ValidationConfig = None,
        regime: RegimeConfig = None
    ):
        self.risk = risk or RiskConfig()
        self.costs = costs or CostConfig()
        self.lstm = lstm or LSTMConfig()
        self.tda = tda or TDAConfig()
        self.strategy = strategy or StrategyConfig()
        self.validation = validation or ValidationConfig()
        self.regime = regime or RegimeConfig()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        from dataclasses import asdict
        return {
            'risk': asdict(self.risk),
            'costs': asdict(self.costs),
            'lstm': asdict(self.lstm),
            'tda': asdict(self.tda),
            'strategy': asdict(self.strategy),
            'validation': asdict(self.validation),
            'regime': asdict(self.regime)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ProductionConfig':
        """Create from dictionary."""
        return cls(
            risk=RiskConfig(**config_dict.get('risk', {})),
            costs=CostConfig(**config_dict.get('costs', {})),
            lstm=LSTMConfig(**config_dict.get('lstm', {})),
            tda=TDAConfig(**config_dict.get('tda', {})),
            strategy=StrategyConfig(**config_dict.get('strategy', {})),
            validation=ValidationConfig(**config_dict.get('validation', {})),
            regime=RegimeConfig(**config_dict.get('regime', {}))
        )
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, path: str) -> 'ProductionConfig':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default production configuration
DEFAULT_CONFIG = ProductionConfig()


# Conservative configuration for live trading
CONSERVATIVE_CONFIG = ProductionConfig(
    risk=RiskConfig(
        kelly_fraction=0.125,  # 1/8 Kelly
        risk_per_trade=0.005,  # 0.5% risk
        max_position_pct=0.05,  # 5% max position
        max_portfolio_heat=0.10  # 10% max heat
    ),
    strategy=StrategyConfig(
        buy_threshold=0.60,  # Higher threshold
        sell_threshold=0.40,  # Wider exit
        min_confidence=0.15  # More confidence required
    )
)


# Aggressive configuration for paper trading testing
AGGRESSIVE_CONFIG = ProductionConfig(
    risk=RiskConfig(
        kelly_fraction=0.75,  # 3/4 Kelly (aggressive)
        risk_per_trade=0.03,  # 3% risk
        max_position_pct=0.20,  # 20% max position
        max_portfolio_heat=0.40  # 40% max heat
    ),
    strategy=StrategyConfig(
        buy_threshold=0.52,  # Lower threshold
        sell_threshold=0.48,  # Tighter exit
        min_confidence=0.05  # Less confidence required
    )
)


def get_config(profile: str = 'default') -> ProductionConfig:
    """Get configuration by profile name.
    
    Args:
        profile: 'default', 'conservative', or 'aggressive'
    
    Returns:
        ProductionConfig instance
    """
    profiles = {
        'default': DEFAULT_CONFIG,
        'conservative': CONSERVATIVE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG
    }
    return profiles.get(profile, DEFAULT_CONFIG)


if __name__ == "__main__":
    # Print default configuration
    import json
    
    print("=" * 70)
    print("PRODUCTION CONFIGURATION")
    print("=" * 70)
    
    config = DEFAULT_CONFIG
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save to file
    config.save_json('results/production_config.json')
    print("\nConfiguration saved to: results/production_config.json")
