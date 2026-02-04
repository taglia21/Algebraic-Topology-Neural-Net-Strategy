# Medallion Math Integration Guide

## Overview
Renaissance Technologies-inspired mathematical foundations now available in `src/medallion_math.py`

## Quick Start

```python
from src.medallion_math import MedallionStrategy
import numpy as np

# Initialize
strategy = MedallionStrategy()

# Analyze market
result = strategy.analyze(
    prices=price_array,  # numpy array of prices
    volumes=volume_array  # optional
)

# Get recommendation
print(f"Strategy: {result['recommended_strategy']}")  # TREND_FOLLOWING, MEAN_REVERSION, or NEUTRAL
print(f"Confidence: {result['strategy_confidence']:.2%}")
print(f"Market Regime: {result['regime']}")  # Bull, Bear, HighVol, Sideways
```

## Components

### 1. Hurst Exponent (Trend vs Mean-Reversion)
```python
from src.medallion_math import HurstExponent

h = HurstExponent.calculate(returns)
if h > 0.55:
    # Trending market - use momentum strategies
    pass
elif h < 0.45:
    # Mean-reverting - use stat arb
    pass
```

### 2. Market Regime Detection (HMM)
```python
from src.medallion_math import MarketRegimeHMM

hmm = MarketRegimeHMM(n_states=4)
hmm.fit(returns, volumes)
regime_id, regime_name, probs = hmm.predict_regime(returns, volumes)

# Adjust position sizing based on regime
if regime_name == 'HighVol':
    position_size *= 0.5  # reduce risk
elif regime_name == 'Bull':
    position_size *= 1.2  # increase exposure
```

### 3. Ornstein-Uhlenbeck Mean Reversion
```python
from src.medallion_math import OrnsteinUhlenbeck

ou = OrnsteinUhlenbeck()
ou.fit(spread_series)  # e.g., pair trading spread
signal = ou.get_signal(current_spread, entry_z=2.0, exit_z=0.5)

if signal['action'] == 'LONG':
    # Enter long when spread is -2 std below mean
    pass
elif signal['action'] == 'SHORT':
    # Enter short when spread is +2 std above mean
    pass
elif signal['action'] == 'EXIT':
    # Exit when spread returns to mean
    pass

print(f"Half-life: {signal['half_life_days']:.1f} days")  # how long to hold
```

### 4. Wavelet Denoising
```python
from src.medallion_math import WaveletDenoiser

denoiser = WaveletDenoiser(wavelet='db4', level=4)
clean_signal = denoiser.denoise(noisy_prices)

# Use clean signal for indicator calculation
rsi = calculate_rsi(clean_signal)  # less false signals
```

### 5. Persistent Homology Turbulence (Crash Warning)
```python
from src.medallion_math import PersistentHomologyTurbulence

turb = PersistentHomologyTurbulence()
turb.fit_baseline(historical_returns_matrix)  # N x M (days x assets)

current_turbulence = turb.get_turbulence(recent_returns_matrix)

if current_turbulence > 3.0:  # high turbulence threshold
    # Reduce all positions - crash risk elevated
    print("⚠️  High market turbulence detected")
```

## Integration with Existing System

### Option 1: Add to `enhanced_trading_engine.py`

```python
from src.medallion_math import MedallionStrategy

class EnhancedTradingEngine:
    def __init__(self):
        self.medallion = MedallionStrategy()
        # ... existing code ...
    
    def generate_signals(self, symbol, data):
        # Get Medallion analysis
        prices = data['close'].values
        volumes = data['volume'].values if 'volume' in data else None
        
        analysis = self.medallion.analyze(prices, volumes)
        
        # Override strategy based on mathematical analysis
        if analysis['regime'] == 'HighVol':
            # Reduce position size in high volatility
            self.risk_multiplier = 0.5
        
        if analysis['recommended_strategy'] == 'TREND_FOLLOWING':
            # Use momentum indicators
            signal = self.calculate_momentum_signal(data)
        elif analysis['recommended_strategy'] == 'MEAN_REVERSION':
            # Use mean reversion
            signal = analysis['ou_signal']
        else:
            # Market is random - reduce exposure
            signal = {'action': 'HOLD'}
        
        return signal
```

### Option 2: Pre-Trade Filter

```python
def should_trade(symbol, prices, volumes):
    """Gate-keeping function using Medallion math"""
    from src.medallion_math import HurstExponent, PersistentHomologyTurbulence
    
    # Check if market is tradeable
    h = HurstExponent.calculate(np.diff(np.log(prices)))
    
    if 0.45 < h < 0.55:
        # Random walk - skip trading
        return False
    
    # Check for crash risk
    turb = PersistentHomologyTurbulence()
    if turb.get_turbulence(returns_matrix) > 3.0:
        # High systemic risk - reduce trading
        return False
    
    return True
```

## Performance Tips

1. **Hurst Exponent**: Recalculate daily, use 100-500 data points
2. **HMM Regime**: Fit weekly, predict daily
3. **O-U Process**: Fit every 4 hours for intraday, daily for swing trading
4. **Wavelet Denoising**: Apply before calculating any technical indicators
5. **Turbulence Index**: Update hourly during market hours

## Example Trading Logic

```python
def medallion_trading_decision(symbol, historical_data):
    from src.medallion_math import MedallionStrategy
    
    ms = MedallionStrategy()
    result = ms.analyze(
        historical_data['close'].values,
        historical_data['volume'].values
    )
    
    # Decision tree
    if result['regime'] == 'HighVol' and result['ou_signal']['z_score'] > 2.5:
        return {
            'action': 'SELL',
            'reason': 'High volatility + overbought',
            'size': 0.5  # reduced size
        }
    
    if result['recommended_strategy'] == 'TREND_FOLLOWING' and result['strategy_confidence'] > 0.7:
        return {
            'action': 'BUY',
            'reason': 'Strong trending regime detected',
            'size': 1.0
        }
    
    if result['ou_signal']['action'] == 'LONG' and result['ou_signal']['half_life_days'] < 30:
        return {
            'action': 'BUY',
            'reason': 'Mean reversion opportunity',
            'size': 0.8,
            'hold_days': result['ou_signal']['half_life_days']
        }
    
    return {'action': 'HOLD'}
```

## Next Steps

1. ✅ Module created and tested
2. ⬜ Add to `requirements.txt`: `hmmlearn>=0.3.0`, `PyWavelets>=1.4.0`
3. ⬜ Import into your main trading engine
4. ⬜ Backtest with historical data
5. ⬜ Paper trade for 1 week to validate
6. ⬜ Deploy to production

## Dependencies

```bash
pip install hmmlearn PyWavelets numpy scipy
```

Already installed in your environment ✅
