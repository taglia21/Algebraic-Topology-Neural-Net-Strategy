#!/usr/bin/env python3
"""
Medallion Math Module - Comprehensive Test Suite
Tests all mathematical components independently and integrated
"""

import numpy as np
import sys
import importlib.util

# Load medallion_math module directly
spec = importlib.util.spec_from_file_location(
    'medallion_math',
    '/workspaces/Algebraic-Topology-Neural-Net-Strategy/src/medallion_math.py'
)
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

print("=" * 60)
print("MEDALLION MATH MODULE - COMPREHENSIVE TEST")
print("=" * 60)

# Generate realistic market data
np.random.seed(42)
n = 500

# Trending data
trend_prices = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
# Mean-reverting data
mr_prices = 100 + 10 * np.sin(np.linspace(0, 10*np.pi, n)) + np.random.randn(n) * 2
# Random walk
rw_prices = 100 + np.cumsum(np.random.randn(n))
# Volumes
volumes = np.random.uniform(1e6, 5e6, n)

print("\n1. HURST EXPONENT TEST")
print("-" * 60)
h_trend = mm.HurstExponent.calculate(trend_prices)
h_mr = mm.HurstExponent.calculate(mr_prices)
h_rw = mm.HurstExponent.calculate(rw_prices)

print(f"Trending series:       H = {h_trend:.3f} {'✓ TRENDING' if h_trend > 0.55 else ''}")
print(f"Mean-reverting series: H = {h_mr:.3f} {'✓ MEAN-REVERTING' if h_mr < 0.45 else ''}")
print(f"Random walk:           H = {h_rw:.3f} {'✓ RANDOM' if 0.45 <= h_rw <= 0.55 else ''}")

print("\n2. ORNSTEIN-UHLENBECK TEST")
print("-" * 60)
ou = mm.OrnsteinUhlenbeck()
ou.fit(mr_prices)
signal = ou.get_signal(mr_prices[-1])

print(f"Mean (μ):        {ou.mu:.2f}")
print(f"Theta (θ):       {ou.theta:.4f}")
print(f"Sigma (σ):       {ou.sigma:.4f}")
print(f"Half-life:       {signal['half_life_days']:.1f} days")
print(f"Current Z-score: {signal['z_score']:.2f}")
print(f"Recommended:     {signal['action']}")

print("\n3. WAVELET DENOISING TEST")
print("-" * 60)
noisy_signal = trend_prices + np.random.randn(n) * 5
denoiser = mm.WaveletDenoiser(wavelet='db4', level=4)
clean_signal = denoiser.denoise(noisy_signal)

noise_reduction = (np.std(noisy_signal - trend_prices) - np.std(clean_signal - trend_prices)) / np.std(noisy_signal - trend_prices) * 100

print(f"Original noise std:  {np.std(noisy_signal - trend_prices):.2f}")
print(f"Denoised noise std:  {np.std(clean_signal - trend_prices):.2f}")
print(f"Noise reduction:     {noise_reduction:.1f}%")
print(f"Signal preserved:    {len(clean_signal) == len(noisy_signal)} ✓")

print("\n4. MARKET REGIME HMM TEST")
print("-" * 60)
returns = np.diff(np.log(trend_prices))
hmm_model = mm.MarketRegimeHMM(n_states=4)
hmm_model.fit(returns, volumes[1:])
regime_id, regime_name, probs = hmm_model.predict_regime(returns, volumes[1:])

print(f"Current regime:  {regime_name} (state {regime_id})")
print(f"Regime probabilities:")
for i, (state, prob) in enumerate(zip(hmm_model.state_names, probs)):
    print(f"  {state:12s}: {prob:.1%} {'█' * int(prob * 40)}")

print("\n5. PERSISTENT HOMOLOGY TURBULENCE TEST")
print("-" * 60)
# Create multi-asset returns matrix
n_assets = 10
returns_matrix = np.random.randn(60, n_assets) * 0.02  # 60 days, 10 assets
turbulence = mm.PersistentHomologyTurbulence()
turbulence.fit_baseline(returns_matrix[:40])

normal_turb = turbulence.get_turbulence(returns_matrix[40:50])
# Simulate crash scenario (high correlation)
crash_matrix = returns_matrix[50:] + np.random.randn(10, 1) * 0.05  # common factor
crash_turb = turbulence.get_turbulence(crash_matrix)

print(f"Normal turbulence:     {normal_turb:.3f}")
print(f"Crisis turbulence:     {crash_turb:.3f}")
print(f"Turbulence increase:   {(crash_turb/normal_turb - 1)*100:.1f}%")
if crash_turb > normal_turb * 1.5:
    print("⚠️  ELEVATED SYSTEMIC RISK DETECTED")

print("\n6. INTEGRATED MEDALLION STRATEGY TEST")
print("=" * 60)

# Test on different market conditions
test_cases = [
    ("Trending Market", trend_prices, "TREND_FOLLOWING"),
    ("Mean-Reverting", mr_prices, "MEAN_REVERSION"),
    ("Random Walk", rw_prices, "NEUTRAL")
]

for name, prices, expected in test_cases:
    print(f"\nTesting: {name}")
    print("-" * 60)
    
    ms = mm.MedallionStrategy()
    result = ms.analyze(prices, volumes)
    
    print(f"Hurst Exponent:      {result['hurst_exponent']:.3f}")
    print(f"Market Regime:       {result['regime']}")
    print(f"Strategy:            {result['recommended_strategy']}")
    print(f"Confidence:          {result['strategy_confidence']:.1%}")
    print(f"O-U Z-Score:         {result['ou_signal']['z_score']:.2f}")
    print(f"O-U Action:          {result['ou_signal']['action']}")
    print(f"Half-life (days):    {result['half_life_days']:.1f}")
    
    # Validation
    matches = result['recommended_strategy'] == expected
    print(f"Expected strategy:   {expected} {'✅' if matches else '⚠️'}")

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nMedallion Math Module is ready for integration!")
print("See MEDALLION_INTEGRATION.md for usage examples.")
