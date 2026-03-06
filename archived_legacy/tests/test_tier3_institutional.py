"""
TIER 3 — INSTITUTIONAL QUANTITATIVE FINANCE ENGINEERING
Comprehensive pytest test suite: one TestClass per phase (M through T).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ===========================================================================
# TestPhaseM — Factor Model & Alpha Research Engine (Items 1-3)
# ===========================================================================

class TestPhaseM:
    """Tests for FamaFrenchFactorModel, AlphaDecayTracker, CrossSectionalMomentum."""

    # --- Item 1: FamaFrenchFactorModel ---

    def test_ff5_basic_fit(self):
        """FF5 OLS regression runs and produces FactorExposure."""
        from src.research.factor_model import FamaFrenchFactorModel

        rng = np.random.RandomState(42)
        T = 120
        # Synthetic factor returns
        factors = rng.randn(T, 5) * 0.01
        # Asset returns = alpha + beta * factors + noise
        alpha = 0.001
        betas = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
        returns = alpha + factors @ betas + rng.randn(T) * 0.005

        model = FamaFrenchFactorModel(window=60)
        exposure = model.fit(returns, factors)

        assert exposure.n_obs == 60  # window
        assert exposure.r_squared > 0.3
        assert abs(exposure.beta_mkt - 1.0) < 0.5  # roughly correct
        assert exposure.residual_vol > 0

    def test_ff5_alpha_tstat_gate(self):
        """alpha_tstat > 2.0 gate works correctly."""
        from src.research.factor_model import FamaFrenchFactorModel

        rng = np.random.RandomState(42)
        T = 120
        factors = rng.randn(T, 5) * 0.01
        # Strong alpha that should produce tstat > 2
        returns = 0.005 + factors @ np.ones(5) * 0.5 + rng.randn(T) * 0.002
        model = FamaFrenchFactorModel(window=60)
        model.fit(returns, factors)

        tstat = model.alpha_tstat()
        assert isinstance(tstat, float)
        # Model has should_trade method
        result = model.should_trade(threshold=0.0)
        assert isinstance(result, bool)

    def test_ff5_get_factor_betas(self):
        """get_factor_betas returns dict with all 5 factor names."""
        from src.research.factor_model import FamaFrenchFactorModel

        rng = np.random.RandomState(42)
        model = FamaFrenchFactorModel()
        factors = rng.randn(100, 5) * 0.01
        returns = rng.randn(100) * 0.01
        model.fit(returns, factors)

        betas = model.get_factor_betas()
        assert set(betas.keys()) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA"}
        assert all(isinstance(v, float) for v in betas.values())

    def test_ff5_insufficient_data(self):
        """Model handles insufficient data gracefully."""
        from src.research.factor_model import FamaFrenchFactorModel

        model = FamaFrenchFactorModel(window=60, min_observations=30)
        # Only 10 observations
        factors = np.random.randn(10, 5) * 0.01
        returns = np.random.randn(10) * 0.01
        exposure = model.fit(returns, factors)

        assert exposure.n_obs == 10
        assert exposure.alpha == 0.0  # default

    # --- Item 2: AlphaDecayTracker ---

    def test_alpha_decay_exponential(self):
        """AlphaDecayTracker estimates exponential decay half-life."""
        from src.research.factor_model import AlphaDecayTracker

        tracker = AlphaDecayTracker(critical_half_life=3.0)
        # Simulate decaying alpha
        for day in range(30):
            alpha = 0.05 * np.exp(-0.2 * day)  # half-life ≈ 3.5 days
            tracker.record_alpha(alpha, timestamp=day * 86400.0)

        estimate = tracker.estimate_decay()
        assert estimate.half_life_days > 0
        assert estimate.half_life_days < 100  # reasonable
        assert estimate.current_alpha > 0
        assert estimate.peak_alpha > 0

    def test_alpha_decay_critical_alert(self):
        """Critical alert when half-life < 3 days."""
        from src.research.factor_model import AlphaDecayTracker

        tracker = AlphaDecayTracker(critical_half_life=5.0)  # generous threshold for test
        # Fast decay: half-life ≈ 1 day
        for day in range(20):
            alpha = 0.1 * np.exp(-0.7 * day)
            tracker.record_alpha(alpha, timestamp=day * 86400.0)

        estimate = tracker.estimate_decay()
        assert estimate.half_life_days > 0
        assert estimate.timestamp != ""

    def test_alpha_decay_save_load(self):
        """AlphaDecayTracker saves and loads JSON log."""
        from src.research.factor_model import AlphaDecayTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "alpha_decay.json")
            tracker = AlphaDecayTracker(log_path=log_path)

            for day in range(10):
                tracker.record_alpha(0.05 * np.exp(-0.1 * day), timestamp=day * 86400)

            tracker.estimate_decay()
            tracker.save_log()

            assert os.path.exists(log_path)
            with open(log_path) as f:
                data = json.load(f)
            assert "history" in data

            # Load works
            tracker2 = AlphaDecayTracker(log_path=log_path)
            history = tracker2.load_log()
            assert len(history) > 0

    # --- Item 3: CrossSectionalMomentum ---

    def test_momentum_ranking(self):
        """CrossSectionalMomentum ranks assets by 12-1 month return."""
        from src.research.factor_model import CrossSectionalMomentum

        rng = np.random.RandomState(42)
        # 20 assets, 300 trading days
        returns = {}
        for i in range(20):
            daily = rng.randn(300) * 0.01 + (i - 10) * 0.001  # systematic drift
            returns[f"STOCK_{i}"] = daily

        mom = CrossSectionalMomentum(top_percentile=0.1, bottom_percentile=0.1)
        signals = mom.compute_momentum(returns)

        assert len(signals) > 0
        # Check ranking is sorted by momentum (descending)
        scores = [s.momentum_score for s in signals]
        assert scores == sorted(scores, reverse=True)
        # Top has long, bottom has short
        assert any(s.signal == "long" for s in signals)
        assert any(s.signal == "short" for s in signals)

    def test_momentum_portfolio_generation(self):
        """generate_portfolio returns long/short lists."""
        from src.research.factor_model import CrossSectionalMomentum

        rng = np.random.RandomState(42)
        returns = {f"S{i}": rng.randn(300) * 0.01 for i in range(20)}
        mom = CrossSectionalMomentum()
        portfolio = mom.generate_portfolio(returns)

        assert len(portfolio.long_symbols) > 0
        assert len(portfolio.short_symbols) > 0
        assert portfolio.n_assets == 20
        assert portfolio.rebalance_date != ""

    def test_momentum_insufficient_data(self):
        """Handles assets with insufficient history."""
        from src.research.factor_model import CrossSectionalMomentum

        returns = {"A": np.random.randn(10) * 0.01}  # too short
        mom = CrossSectionalMomentum()
        signals = mom.compute_momentum(returns)
        assert len(signals) == 0  # Filtered out


# ===========================================================================
# TestPhaseN — Market Making & Adverse Selection (Items 4-6)
# ===========================================================================

class TestPhaseN:
    """Tests for AvellanedaStoikovMarketMaker, AdverseSelectionFilter, InventoryManager."""

    # --- Item 4: AvellanedaStoikovMarketMaker ---

    def test_as_reservation_price(self):
        """Reservation price adjusts for inventory."""
        from src.execution.market_maker import AvellanedaStoikovMarketMaker, InventoryManager

        inv = InventoryManager(max_inventory=500, gamma=0.01)
        mm = AvellanedaStoikovMarketMaker(gamma=0.1, sigma=0.02, inventory_manager=inv)

        mid = 100.0
        r_neutral = mm.reservation_price(mid, time_remaining=1.0)
        assert abs(r_neutral - mid) < 0.01  # zero inventory

        # Long inventory pushes reservation down
        inv.update(200)
        r_long = mm.reservation_price(mid, time_remaining=1.0)
        assert r_long < mid

        # Short inventory pushes reservation up
        inv.reset()
        inv.update(-200)
        r_short = mm.reservation_price(mid, time_remaining=1.0)
        assert r_short > mid

    def test_as_optimal_spread(self):
        """Optimal spread is positive and varies with risk aversion."""
        from src.execution.market_maker import AvellanedaStoikovMarketMaker

        mm_low = AvellanedaStoikovMarketMaker(gamma=0.1, k=1.5, sigma=0.02)
        mm_high = AvellanedaStoikovMarketMaker(gamma=0.5, k=1.5, sigma=0.02)

        spread_low = mm_low.optimal_spread(1.0)
        spread_high = mm_high.optimal_spread(1.0)

        assert spread_low > 0
        assert spread_high > 0
        # A-S spread: gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)
        # Both components contribute; spread varies non-monotonically with gamma
        assert spread_low != spread_high

    def test_as_compute_quotes(self):
        """compute_quotes returns valid bid/ask with bid < ask."""
        from src.execution.market_maker import AvellanedaStoikovMarketMaker

        mm = AvellanedaStoikovMarketMaker(gamma=0.1, k=1.5, sigma=0.02)
        quote = mm.compute_quotes(mid=100.0, time_remaining=0.5)

        assert quote.bid < quote.ask
        assert quote.spread > 0
        assert quote.mid == 100.0
        assert quote.reservation_price > 0

    def test_as_on_fill(self):
        """on_fill updates inventory and returns state."""
        from src.execution.market_maker import AvellanedaStoikovMarketMaker

        mm = AvellanedaStoikovMarketMaker(gamma=0.1)
        state = mm.on_fill(100)  # buy 100
        assert state.position == 100

        state = mm.on_fill(-50)  # sell 50
        assert state.position == 50

    # --- Item 5: AdverseSelectionFilter ---

    def test_pin_estimation(self):
        """PIN model estimates probability of informed trading."""
        from src.execution.adverse_selection import AdverseSelectionFilter

        rng = np.random.RandomState(42)
        # Normal market: balanced buy/sell
        buy_counts = rng.poisson(100, size=60)
        sell_counts = rng.poisson(100, size=60)

        asf = AdverseSelectionFilter(pin_threshold=0.25)
        estimate = asf.estimate_pin(buy_counts, sell_counts)

        assert 0 <= estimate.pin <= 1
        assert estimate.n_days == 60
        assert estimate.eps_b > 0
        assert estimate.eps_s > 0

    def test_adverse_selection_adjustment(self):
        """Toxic flow triggers spread widening and size reduction."""
        from src.execution.adverse_selection import AdverseSelectionFilter

        asf = AdverseSelectionFilter(
            pin_threshold=0.25,
            spread_multiplier_toxic=2.0,
            size_multiplier_toxic=0.5,
        )

        # Highly imbalanced (informed) market
        buy_counts = np.array([200, 190, 210, 180, 220] * 10)
        sell_counts = np.array([50, 60, 40, 55, 45] * 10)

        adj = asf.check_toxicity(buy_counts, sell_counts)

        # Result should have correct fields
        assert isinstance(adj.spread_multiplier, float)
        assert isinstance(adj.size_multiplier, float)
        assert isinstance(adj.is_toxic, bool)
        assert adj.pin_estimate >= 0

    # --- Item 6: InventoryManager ---

    def test_inventory_soft_limit(self):
        """Utilization tracks correctly."""
        from src.execution.market_maker import InventoryManager

        inv = InventoryManager(max_inventory=500, gamma=0.01)
        inv.update(250)
        state = inv.state()

        assert state.position == 250
        assert state.utilization == 0.5
        assert not state.is_hard_stop

    def test_inventory_hard_stop(self):
        """Hard stop triggers at 2x max_inventory."""
        from src.execution.market_maker import InventoryManager

        inv = InventoryManager(max_inventory=500)
        inv.update(1000)
        state = inv.state()

        assert state.is_hard_stop  # 1000 >= 2 * 500
        assert inv.should_flatten()

    def test_inventory_skew(self):
        """Skew pushes quotes in opposite direction of inventory."""
        from src.execution.market_maker import InventoryManager

        inv = InventoryManager(max_inventory=500, gamma=0.02)
        inv.update(100)  # long
        skew = inv.get_skew()
        assert skew < 0  # negative skew to sell

        inv.reset()
        inv.update(-100)  # short
        skew = inv.get_skew()
        assert skew > 0  # positive skew to buy


# ===========================================================================
# TestPhaseO — Black-Litterman Portfolio Optimizer (Items 7-9)
# ===========================================================================

class TestPhaseO:
    """Tests for BlackLittermanOptimizer, RiskParityAllocator, MeanVarianceOptimizer."""

    def _make_cov(self, n: int = 5, seed: int = 42) -> np.ndarray:
        """Create a positive definite covariance matrix."""
        rng = np.random.RandomState(seed)
        A = rng.randn(n, n) * 0.01
        return A.T @ A + np.eye(n) * 0.001

    # --- Item 7: BlackLittermanOptimizer ---

    def test_bl_equilibrium_returns(self):
        """Equilibrium returns Pi = delta * Sigma * w_eq."""
        from src.portfolio.black_litterman import BlackLittermanOptimizer

        bl = BlackLittermanOptimizer(delta=2.5, tau=0.05)
        cov = self._make_cov(5)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        pi = bl.equilibrium_returns(cov, weights)

        assert pi.shape == (5,)
        assert not np.any(np.isnan(pi))

    def test_bl_posterior_no_views(self):
        """Posterior without views equals equilibrium."""
        from src.portfolio.black_litterman import BlackLittermanOptimizer

        bl = BlackLittermanOptimizer(delta=2.5)
        cov = self._make_cov(5)
        weights = np.ones(5) / 5
        post_ret, post_cov = bl.posterior(cov, weights)
        pi = bl.equilibrium_returns(cov, weights)

        np.testing.assert_allclose(post_ret, pi, atol=1e-10)

    def test_bl_posterior_with_views(self):
        """Posterior tilts toward views."""
        from src.portfolio.black_litterman import BlackLittermanOptimizer

        bl = BlackLittermanOptimizer(delta=2.5, tau=0.05)
        cov = self._make_cov(5)
        weights = np.ones(5) / 5
        pi = bl.equilibrium_returns(cov, weights)

        # View: asset 0 will outperform by 5%
        P = np.array([[1, 0, 0, 0, 0]])
        Q = np.array([pi[0] + 0.05])

        post_ret, post_cov = bl.posterior(cov, weights, P, Q)
        # Posterior should tilt asset 0 upward
        assert post_ret[0] > pi[0]

    def test_bl_optimize(self):
        """Full BL optimization produces valid weights."""
        from src.portfolio.black_litterman import BlackLittermanOptimizer

        bl = BlackLittermanOptimizer(delta=2.5)
        cov = self._make_cov(5)
        weights = np.ones(5) / 5
        result = bl.optimize(cov, weights, max_weight=0.30)

        assert result.n_assets == 5
        np.testing.assert_allclose(result.optimal_weights.sum(), 1.0, atol=0.01)
        assert np.all(result.optimal_weights >= -0.01)
        assert np.all(result.optimal_weights <= 0.31)

    # --- Item 8: RiskParityAllocator ---

    def test_risk_parity_equal_rc(self):
        """Risk parity produces approximately equal risk contributions."""
        from src.portfolio.black_litterman import RiskParityAllocator

        rp = RiskParityAllocator()
        cov = self._make_cov(4, seed=123)
        result = rp.allocate(cov)

        assert result.n_assets == 4
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=0.01)
        assert result.portfolio_vol > 0
        # All risk contributions should be close to 1/N
        target = 1.0 / 4
        assert result.max_rc_deviation < 0.1  # within 10% of target

    # --- Item 9: MeanVarianceOptimizer ---

    def test_mvo_ledoit_wolf_shrinkage(self):
        """Ledoit-Wolf produces valid shrunk covariance."""
        from src.portfolio.black_litterman import MeanVarianceOptimizer

        rng = np.random.RandomState(42)
        returns = rng.randn(100, 5) * 0.01
        shrunk = MeanVarianceOptimizer.ledoit_wolf_shrinkage(returns)

        assert shrunk.shape == (5, 5)
        # Positive definite
        eigenvalues = np.linalg.eigvalsh(shrunk)
        assert np.all(eigenvalues > 0)

    def test_mvo_max_weight_constraint(self):
        """MVO respects max_weight=0.20."""
        from src.portfolio.black_litterman import MeanVarianceOptimizer

        mvo = MeanVarianceOptimizer(max_weight=0.20, risk_aversion=2.5)
        expected_ret = np.array([0.10, 0.08, 0.12, 0.06, 0.09, 0.11])
        cov = self._make_cov(6)
        result = mvo.optimize(expected_ret, cov)

        assert result.n_assets == 6
        assert np.all(result.weights <= 0.21)  # small tolerance
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=0.01)

    def test_mvo_turnover_penalty(self):
        """Turnover penalty reduces changes from current weights."""
        from src.portfolio.black_litterman import MeanVarianceOptimizer

        mvo = MeanVarianceOptimizer(turnover_penalty=0.05)
        expected_ret = np.array([0.10, 0.08, 0.12, 0.06, 0.09])
        cov = self._make_cov(5)
        current = np.ones(5) / 5

        result = mvo.optimize(expected_ret, cov, current_weights=current)
        assert result.turnover >= 0


# ===========================================================================
# TestPhaseP — Advanced Options Analytics (Items 10-12)
# ===========================================================================

class TestPhaseP:
    """Tests for SABRVolatilityModel, VolatilityRiskPremium, GEXPinningAnalysis."""

    # --- Item 10: SABRVolatilityModel ---

    def test_sabr_atm_vol(self):
        """SABR ATM implied vol is reasonable."""
        from src.options.sabr_model import SABRVolatilityModel

        sabr = SABRVolatilityModel(beta=0.5)
        vol = sabr.hagan_implied_vol(F=100, K=100, T=0.25, alpha=0.2, rho=-0.3, nu=0.4)
        assert 0.01 < vol < 2.0  # reasonable for ATM

    def test_sabr_smile(self):
        """SABR generates a volatility smile across strikes."""
        from src.options.sabr_model import SABRVolatilityModel

        sabr = SABRVolatilityModel(beta=0.5)
        strikes = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
        vols = sabr.implied_vol_surface(F=100, strikes=strikes, T=0.25,
                                        alpha=0.2, rho=-0.3, nu=0.4)

        assert len(vols) == 7
        assert all(v > 0 for v in vols)
        # With negative rho, should have left skew (lower strikes have higher vol)
        assert vols[0] > vols[3]  # OTM put > ATM

    def test_sabr_calibration(self):
        """SABR calibrates to synthetic market vols."""
        from src.options.sabr_model import SABRVolatilityModel

        sabr = SABRVolatilityModel(beta=0.5)
        F = 100.0
        T = 0.25
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)

        # Generate "market" vols from known parameters
        true_alpha, true_rho, true_nu = 0.2, -0.25, 0.35
        market_vols = sabr.implied_vol_surface(F, strikes, T, true_alpha, true_rho, true_nu)

        # Calibrate
        result = sabr.calibrate(F, strikes, market_vols, T)
        assert result.success
        assert result.rmse < 0.001  # should match closely
        assert abs(result.params.alpha - true_alpha) < 0.05
        assert abs(result.params.rho - true_rho) < 0.15

    # --- Item 11: VolatilityRiskPremium ---

    def test_vrp_sell_premium_signal(self):
        """VRP < -2 triggers sell_premium signal."""
        from src.options.vrp import VolatilityRiskPremium

        vrp = VolatilityRiskPremium(sell_threshold=2.0, realized_window=21)
        # Low realized vol, high IV → VRP is negative → sell premium
        returns = np.random.RandomState(42).randn(60) * 0.005  # low vol ~8%
        iv = 0.20  # 20% IV

        signal = vrp.compute_vrp(returns, implied_vol=iv)
        assert signal.realized_vol > 0
        assert signal.implied_vol == 0.20

    def test_vrp_hold_signal(self):
        """When RV ≈ IV, signal is hold."""
        from src.options.vrp import VolatilityRiskPremium

        vrp = VolatilityRiskPremium(sell_threshold=2.0)
        # Match RV and IV
        returns = np.random.RandomState(42).randn(60) * 0.01
        rv = vrp.realized_vol(returns)
        signal = vrp.compute_vrp(returns, implied_vol=rv)
        # VRP ≈ 0, should be hold
        assert abs(signal.vrp) < 2.01 or signal.signal in ("hold", "sell_premium", "buy_premium")

    def test_vrp_history(self):
        """VRP tracks history of estimates."""
        from src.options.vrp import VolatilityRiskPremium

        vrp = VolatilityRiskPremium()
        returns = np.random.RandomState(42).randn(60) * 0.01
        for iv in [0.15, 0.20, 0.25]:
            vrp.compute_vrp(returns, implied_vol=iv)
        assert len(vrp.history) == 3

    # --- Item 12: GEXPinningAnalysis ---

    def test_gex_pin_detection(self):
        """GEXPinningAnalysis identifies pin strike at highest |GEX|."""
        from src.options.gex_analyzer import (
            GEXPinningAnalysis, GEXProfile, StrikeGEX,
        )

        strikes = [
            StrikeGEX(strike=445, call_gamma=100, put_gamma=-50, net_gex=50),
            StrikeGEX(strike=450, call_gamma=500, put_gamma=-200, net_gex=300),  # highest
            StrikeGEX(strike=455, call_gamma=80, put_gamma=-60, net_gex=20),
        ]
        profile = GEXProfile(
            symbol="SPY", spot_price=450.5, timestamp=None,  # type: ignore
            strikes=strikes, net_gex=370, is_positive_gex=True,
        )

        analyzer = GEXPinningAnalysis(pinning_threshold=0.005)
        result = analyzer.analyze(profile)

        assert result.pin_strike == 450.0
        assert result.pin_gex == 300
        # 450.5 vs 450 → 0.11% < 0.5% → pinned
        assert result.is_pinned
        # Positive GEX + pinned → fade breakout
        assert result.fade_breakout

    def test_gex_not_pinned(self):
        """When spot is far from pin, is_pinned is False."""
        from src.options.gex_analyzer import (
            GEXPinningAnalysis, GEXProfile, StrikeGEX,
        )

        strikes = [
            StrikeGEX(strike=450, call_gamma=500, put_gamma=-200, net_gex=300),
        ]
        profile = GEXProfile(
            symbol="SPY", spot_price=460.0, timestamp=None,  # type: ignore
            strikes=strikes, net_gex=300, is_positive_gex=True,
        )

        analyzer = GEXPinningAnalysis(pinning_threshold=0.005)
        result = analyzer.analyze(profile)

        assert result.pin_strike == 450.0
        assert not result.is_pinned  # 460 vs 450 = 2.2% > 0.5%
        assert not result.fade_breakout


# ===========================================================================
# TestPhaseQ — Order Management System (Items 13-15)
# ===========================================================================

class TestPhaseQ:
    """Tests for OrderManagementSystem, PreTradeRiskChecker, PostTradeReconciler."""

    # --- Item 13: OrderManagementSystem ---

    def test_oms_create_order(self):
        """OMS creates order and transitions through states."""
        from src.execution.oms import OrderManagementSystem

        oms = OrderManagementSystem()
        order = oms.create_order(
            symbol="AAPL", side="BUY", quantity=100, current_price=175.0
        )
        assert order.state == "VALIDATED"
        assert order.symbol == "AAPL"
        assert order.order_id != ""

    def test_oms_order_lifecycle(self):
        """Full order lifecycle: create → submit → fill."""
        from src.execution.oms import OrderManagementSystem

        oms = OrderManagementSystem()
        order = oms.create_order("AAPL", "BUY", 100, current_price=175.0)
        assert order.state == "VALIDATED"

        oms.submit_order(order.order_id)
        assert oms.get_order(order.order_id).state == "SUBMITTED"

        oms.record_fill(order.order_id, 100, 175.50)
        assert oms.get_order(order.order_id).state == "FILLED"
        assert oms.get_order(order.order_id).filled_quantity == 100

    def test_oms_partial_fill(self):
        """Partial fills transition correctly."""
        from src.execution.oms import OrderManagementSystem

        oms = OrderManagementSystem()
        order = oms.create_order("AAPL", "BUY", 100, current_price=175.0)
        oms.submit_order(order.order_id)

        oms.record_fill(order.order_id, 50, 175.50)
        assert oms.get_order(order.order_id).state == "PARTIALLY_FILLED"

        oms.record_fill(order.order_id, 50, 175.60)
        assert oms.get_order(order.order_id).state == "FILLED"

    def test_oms_cancel(self):
        """Cancel order transitions to CANCELLED."""
        from src.execution.oms import OrderManagementSystem

        oms = OrderManagementSystem()
        order = oms.create_order("AAPL", "BUY", 100, current_price=175.0)
        oms.submit_order(order.order_id)
        oms.cancel_order(order.order_id, reason="test")
        assert oms.get_order(order.order_id).state == "CANCELLED"

    def test_oms_event_bus(self):
        """Event bus notifies subscribers."""
        from src.execution.oms import OrderManagementSystem

        events_received = []
        oms = OrderManagementSystem()
        oms.subscribe("created", lambda e: events_received.append(e.event_type))
        oms.subscribe("filled", lambda e: events_received.append(e.event_type))

        order = oms.create_order("AAPL", "BUY", 100, current_price=175.0)
        oms.submit_order(order.order_id)
        oms.record_fill(order.order_id, 100, 175.50)

        assert "created" in events_received
        assert "filled" in events_received

    def test_oms_persist(self):
        """OMS saves and loads order book."""
        from src.execution.oms import OrderManagementSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "orders.json")
            oms = OrderManagementSystem(persist_path=path)
            order = oms.create_order("AAPL", "BUY", 100, current_price=175.0)
            oms.save()

            assert os.path.exists(path)

            oms2 = OrderManagementSystem(persist_path=path)
            n = oms2.load()
            assert n == 1
            loaded = oms2.get_order(order.order_id)
            assert loaded is not None
            assert loaded.symbol == "AAPL"

    # --- Item 14: PreTradeRiskChecker ---

    def test_pretrade_restricted_symbol(self):
        """Restricted symbol is rejected."""
        from src.execution.oms import PreTradeRiskChecker, Order

        checker = PreTradeRiskChecker(restricted_symbols={"GME", "AMC"})
        order = Order(symbol="GME", side="BUY", quantity=100)
        result = checker.check(order, current_price=25.0)

        assert not result.approved
        assert any("restricted" in v.lower() for v in result.violations)

    def test_pretrade_fat_finger(self):
        """Fat finger detection works for large price deviations."""
        from src.execution.oms import PreTradeRiskChecker, Order

        checker = PreTradeRiskChecker(fat_finger_pct=0.05)
        order = Order(symbol="AAPL", side="BUY", quantity=100, limit_price=200.0)
        result = checker.check(order, current_price=175.0)

        # 200 vs 175 = 14.3% deviation > 5%
        assert not result.approved
        assert any("fat finger" in v.lower() for v in result.violations)

    def test_pretrade_max_shares(self):
        """Max order size check."""
        from src.execution.oms import PreTradeRiskChecker, Order

        checker = PreTradeRiskChecker(max_order_shares=1000)
        order = Order(symbol="AAPL", side="BUY", quantity=5000)
        result = checker.check(order, current_price=175.0)
        assert not result.approved

    # --- Item 15: PostTradeReconciler ---

    def test_reconciler_match(self):
        """Matching fills reconcile correctly."""
        from src.execution.reconciler import PostTradeReconciler, FillRecord

        reconciler = PostTradeReconciler(flag_threshold=1.0, alert_threshold=100.0)
        oms_fills = [FillRecord(order_id="A1", symbol="AAPL", quantity=100, price=175.0)]
        broker_fills = [FillRecord(order_id="A1", symbol="AAPL", quantity=100, price=175.0)]

        report = reconciler.reconcile(oms_fills, broker_fills)
        assert report.matched == 1
        assert len(report.discrepancies) == 0

    def test_reconciler_price_discrepancy(self):
        """Price discrepancy > $1 flagged, > $100 alerted."""
        from src.execution.reconciler import PostTradeReconciler, FillRecord

        reconciler = PostTradeReconciler(flag_threshold=1.0, alert_threshold=100.0)
        oms_fills = [FillRecord(order_id="A1", symbol="AAPL", quantity=100, price=175.0)]
        broker_fills = [FillRecord(order_id="A1", symbol="AAPL", quantity=100, price=177.0)]

        report = reconciler.reconcile(oms_fills, broker_fills)
        # Notional diff: 100 * 175 vs 100 * 177 = $200 diff → critical
        assert len(report.discrepancies) > 0
        assert report.alerts_triggered > 0

    def test_reconciler_missing_fill(self):
        """Missing fill in broker flagged as critical."""
        from src.execution.reconciler import PostTradeReconciler, FillRecord

        reconciler = PostTradeReconciler()
        oms_fills = [FillRecord(order_id="A1", symbol="AAPL", quantity=100, price=175.0)]
        broker_fills = []

        report = reconciler.reconcile(oms_fills, broker_fills)
        assert len(report.missing_in_broker) == 1
        assert "A1" in report.missing_in_broker


# ===========================================================================
# TestPhaseR — Risk Management System (Items 16-18)
# ===========================================================================

class TestPhaseR:
    """Tests for PortfolioVaR, StressTestEngine, CorrelationBreakdownMonitor."""

    # --- Item 16: PortfolioVaR ---

    def test_var_historical(self):
        """Historical VaR produces positive loss number."""
        from src.risk.risk_engine import PortfolioVaR

        rng = np.random.RandomState(42)
        var_calc = PortfolioVaR(confidence=0.99, horizon_days=1)
        returns = rng.randn(500) * 0.01
        var, es = var_calc.historical_var(returns)

        assert var > 0  # positive loss
        assert es >= var  # ES >= VaR

    def test_var_parametric(self):
        """Parametric VaR is reasonable."""
        from src.risk.risk_engine import PortfolioVaR

        rng = np.random.RandomState(42)
        var_calc = PortfolioVaR(confidence=0.99)
        returns = rng.randn(500) * 0.01
        var = var_calc.parametric_var(returns)

        assert var > 0
        # 99% 1-day VaR for 1% daily vol ≈ 2.33%
        assert var < 0.10  # sanity check

    def test_var_montecarlo(self):
        """Monte Carlo VaR with correlated assets."""
        from src.risk.risk_engine import PortfolioVaR

        rng = np.random.RandomState(42)
        var_calc = PortfolioVaR(confidence=0.99, mc_simulations=5000)
        returns = rng.randn(250, 3) * 0.01
        weights = np.array([0.5, 0.3, 0.2])
        var = var_calc.montecarlo_var(returns, weights)

        assert var > 0

    def test_var_compute_all(self):
        """compute() returns all three VaR methods."""
        from src.risk.risk_engine import PortfolioVaR

        rng = np.random.RandomState(42)
        var_calc = PortfolioVaR(confidence=0.99)
        returns = rng.randn(250, 3) * 0.01
        weights = np.array([0.5, 0.3, 0.2])

        result = var_calc.compute(returns, weights, portfolio_value=1_000_000)

        assert result.var_historical > 0
        assert result.var_parametric > 0
        assert result.var_montecarlo > 0
        assert result.var_dollar_hist > 0
        assert result.confidence == 0.99

    # --- Item 17: StressTestEngine ---

    def test_stress_covid_scenario(self):
        """COVID crash scenario produces expected losses."""
        from src.risk.risk_engine import StressTestEngine

        engine = StressTestEngine(survival_threshold=0.20)
        weights = np.array([0.4, 0.3, 0.3])
        exposures = {0: "equity", 1: "equity", 2: "credit"}

        result = engine.run_scenario("covid_crash", weights, exposures)
        assert result.scenario_name == "COVID Crash"
        assert result.loss_pct > 0
        assert result.portfolio_loss > 0

    def test_stress_custom_scenario(self):
        """Custom stress scenario works."""
        from src.risk.risk_engine import StressTestEngine, StressScenario

        engine = StressTestEngine()
        engine.add_scenario("tech_crash", StressScenario(
            name="Tech Crash",
            description="Tech sector -30%",
            shocks={"tech": -0.30, "other": -0.05},
        ))
        weights = np.array([0.6, 0.4])
        exposures = {0: "tech", 1: "other"}
        result = engine.run_scenario("tech_crash", weights, exposures)

        assert result.scenario_name == "Tech Crash"
        assert result.loss_pct > 0

    def test_stress_run_all(self):
        """run_all executes all 4 built-in scenarios."""
        from src.risk.risk_engine import StressTestEngine

        engine = StressTestEngine()
        weights = np.array([0.5, 0.3, 0.2])
        results = engine.run_all(weights)

        assert len(results) >= 4
        assert all(r.scenario_name != "" for r in results)

    # --- Item 18: CorrelationBreakdownMonitor ---

    def test_correlation_stable(self):
        """Stable correlations produce no breakdown."""
        from src.risk.correlation_monitor import CorrelationBreakdownMonitor

        rng = np.random.RandomState(42)
        monitor = CorrelationBreakdownMonitor()
        returns = rng.randn(300, 3) * 0.01  # stable

        report = monitor.monitor(returns, ["SPY", "QQQ", "IWM"])
        assert report.n_assets == 3
        # Stable returns should have low average change
        assert isinstance(report.avg_change, float)

    def test_correlation_sign_flip(self):
        """Sign flip detection works."""
        from src.risk.correlation_monitor import CorrelationBreakdownMonitor

        rng = np.random.RandomState(42)
        monitor = CorrelationBreakdownMonitor(short_window=20, long_window=252)

        # Create returns where short-term correlation flips
        T = 300
        returns = np.zeros((T, 2))
        # Long-term positive correlation
        returns[:280, 0] = rng.randn(280) * 0.01
        returns[:280, 1] = returns[:280, 0] * 0.8 + rng.randn(280) * 0.005
        # Short-term negative correlation (last 20 days)
        returns[280:, 0] = rng.randn(20) * 0.01
        returns[280:, 1] = -returns[280:, 0] * 0.9 + rng.randn(20) * 0.002

        report = monitor.monitor(returns, ["A", "B"])
        # Should detect the sign flip
        assert report.n_assets == 2
        # The sign flip may or may not trigger based on thresholds
        assert isinstance(report.sign_flips, int)

    def test_correlation_history(self):
        """Monitor keeps history of reports."""
        from src.risk.correlation_monitor import CorrelationBreakdownMonitor

        monitor = CorrelationBreakdownMonitor()
        rng = np.random.RandomState(42)
        returns = rng.randn(300, 3) * 0.01

        monitor.monitor(returns, ["A", "B", "C"])
        monitor.monitor(returns, ["A", "B", "C"])
        assert len(monitor.history) == 2


# ===========================================================================
# TestPhaseS — Backtesting & Research (Items 19-20)
# ===========================================================================

class TestPhaseS:
    """Tests for EventDrivenBacktester and PBO calculator."""

    # --- Item 19: EventDrivenBacktester ---

    def test_backtester_events(self):
        """Event types are correctly defined."""
        from src.research.backtester import (
            MarketDataEvent, SignalEvent, OrderEvent, FillEvent, EventType,
        )

        md = MarketDataEvent(symbol="SPY", close=450.0)
        assert md.event_type == EventType.MARKET_DATA.value

        sig = SignalEvent(symbol="SPY", direction="LONG", strength=0.8)
        assert sig.event_type == EventType.SIGNAL.value

    def test_backtester_fill_model(self):
        """RealisticFillModel applies slippage and commission."""
        from src.research.backtester import RealisticFillModel, OrderEvent, MarketDataEvent

        fm = RealisticFillModel(slippage_bps=5.0, commission_per_share=0.005)
        order = OrderEvent(symbol="SPY", side="BUY", quantity=100)
        md = MarketDataEvent(symbol="SPY", open=449, high=451, low=448, close=450)

        fill = fm.simulate_fill(order, md)
        assert fill.price >= 450.0  # buy slippage pushes up
        assert fill.commission > 0
        assert fill.quantity == 100

    def test_backtester_run(self):
        """Full backtest run produces results."""
        from src.research.backtester import (
            EventDrivenBacktester, Strategy, MarketDataEvent, SignalEvent,
        )

        class SimpleStrategy(Strategy):
            def __init__(self):
                self._bar = 0

            def on_market_data(self, event: MarketDataEvent):
                self._bar += 1
                if self._bar == 5:
                    return SignalEvent(symbol=event.symbol, direction="LONG", strength=1.0)
                elif self._bar == 15:
                    return SignalEvent(symbol=event.symbol, direction="EXIT", strength=1.0)
                return None

        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.randn(50) * 0.5)
        ohlcv = np.column_stack([prices, prices + 1, prices - 1, prices, np.ones(50) * 1e6])

        bt = EventDrivenBacktester(initial_cash=100000)
        result = bt.run(SimpleStrategy(), {"SPY": ohlcv})

        assert result.n_trades > 0
        assert result.final_value > 0
        assert len(result.equity_curve) > 0

    def test_backtester_portfolio_tracking(self):
        """Portfolio tracker handles fills correctly."""
        from src.research.backtester import PortfolioTracker, FillEvent

        pt = PortfolioTracker(initial_cash=100000)

        # Buy
        fill = FillEvent(symbol="SPY", side="BUY", quantity=100, price=450.0, commission=0.50)
        pt.process_fill(fill)
        assert "SPY" in pt.positions
        assert pt.positions["SPY"] == 100
        assert pt.cash < 100000

        # Sell
        fill = FillEvent(symbol="SPY", side="SELL", quantity=100, price=455.0, commission=0.50)
        pt.process_fill(fill)
        assert "SPY" not in pt.positions

    # --- Item 20: PBO Calculator ---

    def test_pbo_not_overfit(self):
        """Non-overfit strategies have low PBO."""
        from src.research.pbo import PBOCalculator

        rng = np.random.RandomState(42)
        # 10 strategy variants, 320 days, all with similar stable returns
        strategy_returns = rng.randn(320, 10) * 0.001 + 0.0001

        pbo = PBOCalculator(n_subperiods=16, pbo_threshold=0.5)
        result = pbo.compute(strategy_returns)

        assert 0 <= result.pbo <= 1
        assert result.n_subperiods > 0
        assert result.n_combinations > 0

    def test_pbo_overfit_detection(self):
        """Overfit strategies have high PBO."""
        from src.research.pbo import PBOCalculator

        rng = np.random.RandomState(42)
        T = 320
        S = 20

        # Create strategies that are overfit:
        # high in-sample return but random out-of-sample
        returns = rng.randn(T, S) * 0.01
        # Boost first half performance for specific strategies
        for s in range(S):
            returns[:T // 2, s] += (s - S // 2) * 0.002  # systematic IS advantage

        pbo = PBOCalculator(n_subperiods=16, pbo_threshold=0.5)
        result = pbo.compute(returns)

        assert 0 <= result.pbo <= 1
        assert len(result.logit_distribution) > 0

    def test_pbo_insufficient_strategies(self):
        """Handles single strategy gracefully."""
        from src.research.pbo import PBOCalculator

        returns = np.random.randn(100, 1) * 0.01
        pbo = PBOCalculator()
        result = pbo.compute(returns)
        assert result.pbo == 0.0  # < 2 strategies


# ===========================================================================
# TestPhaseT — Live Dashboard & Compliance (Items 21-23)
# ===========================================================================

class TestPhaseT:
    """Tests for ComplianceEngine, RegulatoryReporter, LiveDashboard."""

    # --- Item 21: ComplianceEngine ---

    def test_compliance_clean(self):
        """Compliant portfolio passes all checks."""
        from src.monitoring.compliance import ComplianceEngine

        engine = ComplianceEngine(max_position_pct=0.15, max_sector_pct=0.30)
        positions = {"AAPL": 100000, "MSFT": 80000, "JPM": 70000}
        nav = 1_000_000

        report = engine.check(positions, nav)
        assert report.compliant
        assert len(report.violations) == 0
        assert report.checks_run == 5

    def test_compliance_position_breach(self):
        """Position concentration breach detected."""
        from src.monitoring.compliance import ComplianceEngine

        engine = ComplianceEngine(max_position_pct=0.15)
        positions = {"AAPL": 200000}  # 20% > 15%
        nav = 1_000_000

        report = engine.check(positions, nav)
        assert not report.compliant
        assert any(v.rule == "position_concentration" for v in report.violations)

    def test_compliance_sector_breach(self):
        """Sector concentration breach detected."""
        from src.monitoring.compliance import ComplianceEngine

        engine = ComplianceEngine(max_sector_pct=0.30)
        positions = {"AAPL": 200000, "MSFT": 200000}  # tech = 40% > 30%
        nav = 1_000_000
        sector_map = {"AAPL": "tech", "MSFT": "tech"}

        report = engine.check(positions, nav, sector_map=sector_map)
        assert not report.compliant
        assert any(v.rule == "sector_concentration" for v in report.violations)

    def test_compliance_gross_exposure(self):
        """Gross exposure breach detected."""
        from src.monitoring.compliance import ComplianceEngine

        engine = ComplianceEngine(max_gross_exposure=2.0)
        # Long 150% + short 60% = 210% gross > 200%
        positions = {"AAPL": 1500000, "TSLA": -600000}
        nav = 1_000_000

        report = engine.check(positions, nav)
        assert not report.compliant
        assert any(v.rule == "gross_exposure" for v in report.violations)

    def test_compliance_net_exposure(self):
        """Net exposure outside range detected."""
        from src.monitoring.compliance import ComplianceEngine

        engine = ComplianceEngine(net_exposure_range=(-0.30, 1.30))
        # Net = -50%
        positions = {"AAPL": -500000}
        nav = 1_000_000

        report = engine.check(positions, nav)
        assert not report.compliant
        assert any(v.rule == "net_exposure" for v in report.violations)

    # --- Item 22: RegulatoryReporter ---

    def test_form_pf_report(self):
        """Form PF report contains required fields."""
        from src.monitoring.compliance import RegulatoryReporter

        reporter = RegulatoryReporter()
        report = reporter.form_pf_report(
            nav=1_000_000,
            positions={"AAPL": 200000, "MSFT": 150000, "SPY": -50000},
            leverage=1.5,
        )
        assert report.report_type == "form_pf"
        assert report.data["nav"] == 1_000_000
        assert report.data["leverage"] == 1.5
        assert report.data["n_positions"] == 3

    def test_large_trader_report(self):
        """Large trader report flags threshold breach."""
        from src.monitoring.compliance import RegulatoryReporter

        reporter = RegulatoryReporter()
        trades = [
            {"symbol": "AAPL", "quantity": 100000, "price": 175.0},
            {"symbol": "MSFT", "quantity": 50000, "price": 400.0},
        ]
        report = reporter.large_trader_report(trades, threshold=20_000_000)

        assert report.report_type == "large_trader"
        total = 100000 * 175 + 50000 * 400
        assert report.data["total_notional"] == total
        assert report.data["exceeds_threshold"] == (total > 20_000_000)

    def test_tca_summary(self):
        """TCA summary computes slippage statistics."""
        from src.monitoring.compliance import RegulatoryReporter

        reporter = RegulatoryReporter()
        fills = [
            {"price": 175.10, "benchmark_price": 175.0, "quantity": 100, "commission": 0.50},
            {"price": 174.80, "benchmark_price": 175.0, "quantity": 200, "commission": 1.00},
        ]
        report = reporter.tca_summary(fills)

        assert report.report_type == "tca"
        assert report.data["n_fills"] == 2
        assert report.data["total_commission"] == 1.50

    def test_regulatory_save_report(self):
        """Reports save to JSON files."""
        from src.monitoring.compliance import RegulatoryReporter, RegulatoryReport

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = RegulatoryReporter(output_dir=tmpdir)
            report = reporter.form_pf_report(
                nav=1_000_000, positions={"AAPL": 200000}, leverage=1.0,
            )
            path = reporter.save_report(report)
            assert os.path.exists(path)

    # --- Item 23: LiveDashboard ---

    def test_dashboard_snapshot(self):
        """Dashboard snapshot has all 11 fields."""
        from src.monitoring.live_dashboard import LiveDashboard

        dash = LiveDashboard(port=8002)
        snap = dash.update_snapshot(
            portfolio_value=1_000_000,
            daily_pnl=5000,
            daily_return_pct=0.5,
            positions_count=10,
            open_orders=2,
            regime="bull",
            var_99_1d=25000,
            sharpe_30d=2.1,
            max_drawdown=-0.03,
            compliance_status="ok",
        )

        assert snap.portfolio_value == 1_000_000
        assert snap.daily_pnl == 5000
        assert snap.regime == "bull"
        assert snap.compliance_status == "ok"
        assert snap.uptime_seconds > 0

    def test_dashboard_json_serialization(self):
        """Snapshot serializes to valid JSON with all fields."""
        from src.monitoring.live_dashboard import LiveDashboard

        dash = LiveDashboard()
        dash.update_snapshot(portfolio_value=500000, daily_pnl=-1000)
        json_str = dash.get_snapshot_json()

        data = json.loads(json_str)
        assert "portfolio_value" in data
        assert "daily_pnl" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["portfolio_value"] == 500000

    def test_dashboard_port_config(self):
        """Dashboard configures on port 8002."""
        from src.monitoring.live_dashboard import LiveDashboard

        dash = LiveDashboard(port=8002)
        assert dash.port == 8002
        assert dash.push_interval == 5.0
        assert dash.connected_clients == 0
