"""
etf
===
ETF-only tactical asset-allocation engine for Interactive Brokers.

This is a self-contained alpha program that trades **exclusively liquid ETFs**.
It does NOT modify or depend on the equities (Alpaca) or VRP (options) engines.

Design philosophy (anti-overfitting, evidence-first)
----------------------------------------------------
The engine combines only the most *durable, out-of-sample-robust* published
edges, deliberately avoiding fragile, over-parameterised signals:

1. Time-series momentum / trend filter
   (Moskowitz, Ooi & Pedersen 2012; Faber 2007 "A Quantitative Approach to
   Tactical Asset Allocation"): only hold an asset when it is in an uptrend.
   This is the single largest drawdown reducer.

2. Cross-sectional momentum
   (Jegadeesh & Titman 1993; Antonacci "Dual Momentum"): own the strongest
   assets, rotate out of the weakest.

3. Inverse-volatility / risk-parity weighting
   (Asness, Frazzini & Pedersen 2012): size positions by risk, not by guess,
   so no single volatile ETF dominates the portfolio.

4. Portfolio volatility targeting
   (Barroso & Santa-Clara 2015 "Momentum has its moments"): scale gross
   exposure to a constant risk budget; cut exposure when vol spikes.

5. Drawdown de-risking overlay: progressively reduce exposure during equity
   drawdowns, holding cash until the trend repairs.

Every parameter is exposed in ``etf.config`` with a documented rationale.
"""

from etf.config import ETFConfig, get_default_config

__all__ = ["ETFConfig", "get_default_config"]
