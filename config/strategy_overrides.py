#!/usr/bin/env python3
"""
Strategy Performance Overrides
==============================
Emergency performance fix configuration based on ablation study results.

Key Findings from V2_ENHANCEMENT_REPORT.md:
- TDA: -0.112 Sharpe contribution (HARMFUL)
- Risk Parity: -0.219 Sharpe contribution (HARMFUL)
- Combined removal expected: +0.33 Sharpe improvement

Signal Threshold Analysis from multiasset_robustness_report.json:
- Old thresholds (0.52/0.48): 0% buy, 94% sell - severely imbalanced
- New thresholds (0.55/0.45): Expected balanced signal generation

Asset Analysis:
- QQQ Sharpe: 0.39 (weakest, drags down portfolio)
- SPY Sharpe: 0.85 (keep)
- XLF Sharpe: 0.80 (keep)
- XLK Sharpe: 0.65 (keep)
- IWM Sharpe: 0.56 (keep, borderline)

Created: 2026-02-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class StrategyOverrides:
    """
    Configuration overrides for strategy performance fixes.
    
    These settings override defaults in other config modules to improve
    the Sharpe ratio from ~0.38 to target >0.5.
    """
    
    # =========================================================================
    # FEATURE TOGGLES - Disable harmful components
    # =========================================================================
    
    # TDA (Topological Data Analysis) - Hurts Sharpe by -0.112
    enable_tda: bool = False
    enable_persistent_laplacian: bool = False
    enable_betti_numbers: bool = False
    
    # Risk Parity Weighting - Hurts Sharpe by -0.219
    enable_risk_parity: bool = False
    use_inverse_volatility_weighting: bool = False
    
    # SAC Position Sizing - Hurts Sharpe by -0.019
    enable_sac_sizing: bool = False
    
    # =========================================================================
    # FEATURES TO KEEP ENABLED - These help performance
    # =========================================================================
    
    # Ensemble Regime Detection - +0.219 Sharpe contribution
    enable_ensemble_regime: bool = True
    
    # Transformer Predictor - +0.088 Sharpe contribution
    enable_transformer: bool = True
    
    # Enhanced Momentum - +0.003 Sharpe contribution
    enable_enhanced_momentum: bool = True
    
    # =========================================================================
    # SIGNAL THRESHOLDS - Recalibrated for balanced signals
    # =========================================================================
    
    # Old: 0.52/0.48 produced 0% buy / 94% sell
    # New: 0.55/0.45 with neutral zone for balanced signals
    nn_buy_threshold: float = 0.55
    nn_sell_threshold: float = 0.45
    use_neutral_zone: bool = True
    
    # Confidence thresholds
    min_confidence: float = 0.55  # Slightly lower to allow more signals
    signal_threshold: float = 0.55
    
    # =========================================================================
    # UNIVERSE EXCLUSIONS - Remove weak assets
    # =========================================================================
    
    # Assets to exclude (low Sharpe, drag on portfolio)
    excluded_tickers: Set[str] = field(default_factory=lambda: {'QQQ'})
    
    # Minimum per-asset Sharpe to include
    min_asset_sharpe: float = 0.45
    
    # =========================================================================
    # POSITION SIZING OVERRIDES
    # =========================================================================
    
    # Use momentum-weighted allocation instead of risk parity
    allocation_method: str = "momentum_weighted"  # Options: momentum_weighted, equal, signal_strength
    
    # Position limits
    max_position_pct: float = 0.10  # Slightly higher for concentrated bets
    min_position_pct: float = 0.02
    max_cash_pct: float = 0.30  # Allow more cash when signals weak
    
    # =========================================================================
    # SIGNAL COMBINATION WEIGHTS
    # =========================================================================
    
    # Old: 0.4 NN + 0.3 persistence + 0.3 trend (TDA dilutes signal)
    # New: Pure momentum/NN focus
    nn_weight: float = 0.70
    momentum_weight: float = 0.25
    regime_weight: float = 0.05
    tda_weight: float = 0.00  # Disabled
    
    def get_active_features(self) -> List[str]:
        """Get list of enabled features."""
        features = []
        if self.enable_ensemble_regime:
            features.append("ensemble_regime")
        if self.enable_transformer:
            features.append("transformer")
        if self.enable_enhanced_momentum:
            features.append("enhanced_momentum")
        if self.enable_tda:
            features.append("tda")
        if self.enable_risk_parity:
            features.append("risk_parity")
        return features
    
    def get_disabled_features(self) -> List[str]:
        """Get list of disabled features."""
        disabled = []
        if not self.enable_tda:
            disabled.append("tda")
        if not self.enable_risk_parity:
            disabled.append("risk_parity")
        if not self.enable_sac_sizing:
            disabled.append("sac_sizing")
        return disabled
    
    def should_include_ticker(self, ticker: str) -> bool:
        """Check if ticker should be included in universe."""
        return ticker.upper() not in self.excluded_tickers
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            # Feature toggles
            "enable_tda": self.enable_tda,
            "enable_risk_parity": self.enable_risk_parity,
            "enable_sac_sizing": self.enable_sac_sizing,
            "enable_ensemble_regime": self.enable_ensemble_regime,
            "enable_transformer": self.enable_transformer,
            "enable_enhanced_momentum": self.enable_enhanced_momentum,
            
            # Thresholds
            "nn_buy_threshold": self.nn_buy_threshold,
            "nn_sell_threshold": self.nn_sell_threshold,
            "use_neutral_zone": self.use_neutral_zone,
            "min_confidence": self.min_confidence,
            
            # Universe
            "excluded_tickers": list(self.excluded_tickers),
            "min_asset_sharpe": self.min_asset_sharpe,
            
            # Allocation
            "allocation_method": self.allocation_method,
            "max_position_pct": self.max_position_pct,
            
            # Signal weights
            "nn_weight": self.nn_weight,
            "momentum_weight": self.momentum_weight,
            "tda_weight": self.tda_weight,
        }


# Global instance for easy import
STRATEGY_OVERRIDES = StrategyOverrides()


def get_overrides() -> StrategyOverrides:
    """Get the global strategy overrides instance."""
    return STRATEGY_OVERRIDES


def apply_overrides_to_config(config: Dict) -> Dict:
    """Apply overrides to any config dictionary."""
    overrides = get_overrides()
    
    # Apply threshold overrides
    if 'nn_buy_threshold' in config or 'thresholds' in config:
        config['nn_buy_threshold'] = overrides.nn_buy_threshold
        config['nn_sell_threshold'] = overrides.nn_sell_threshold
    
    # Apply feature flags
    config['enable_tda'] = overrides.enable_tda
    config['enable_risk_parity'] = overrides.enable_risk_parity
    config['enable_sac_sizing'] = overrides.enable_sac_sizing
    
    return config


if __name__ == "__main__":
    # Print current configuration
    overrides = get_overrides()
    print("=" * 60)
    print("STRATEGY OVERRIDES - Performance Fix Configuration")
    print("=" * 60)
    
    print("\n[DISABLED FEATURES - These hurt performance]")
    for feature in overrides.get_disabled_features():
        print(f"  ❌ {feature}")
    
    print("\n[ENABLED FEATURES - These help performance]")
    for feature in overrides.get_active_features():
        print(f"  ✅ {feature}")
    
    print("\n[SIGNAL THRESHOLDS]")
    print(f"  Buy threshold:  {overrides.nn_buy_threshold}")
    print(f"  Sell threshold: {overrides.nn_sell_threshold}")
    print(f"  Neutral zone:   {overrides.use_neutral_zone}")
    
    print("\n[EXCLUDED TICKERS]")
    for ticker in overrides.excluded_tickers:
        print(f"  ❌ {ticker}")
    
    print("\n[SIGNAL COMBINATION WEIGHTS]")
    print(f"  NN:       {overrides.nn_weight:.0%}")
    print(f"  Momentum: {overrides.momentum_weight:.0%}")
    print(f"  TDA:      {overrides.tda_weight:.0%}")
    
    print("\n" + "=" * 60)
