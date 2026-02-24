"""
Phase R — Risk Management System.

Item 16: PortfolioVaR — Historical, Parametric, Monte Carlo VaR, 99% 1-day.
Item 17: StressTestEngine — COVID, rate shock, flash crash, custom scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item 16 — PortfolioVaR
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """Value-at-Risk computation result."""
    var_historical: float = 0.0
    var_parametric: float = 0.0
    var_montecarlo: float = 0.0
    es_historical: float = 0.0     # Expected Shortfall (CVaR)
    confidence: float = 0.99
    horizon_days: int = 1
    portfolio_value: float = 0.0
    # Dollar VaR
    var_dollar_hist: float = 0.0
    var_dollar_param: float = 0.0
    var_dollar_mc: float = 0.0


class PortfolioVaR:
    """Portfolio Value-at-Risk calculator.

    Three methods:
      1. Historical Simulation — use empirical return distribution.
      2. Parametric (Gaussian) — assume normal returns.
      3. Monte Carlo — simulate correlated returns.

    Default: 99% 1-day VaR.
    """

    def __init__(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        mc_simulations: int = 10000,
    ):
        self.confidence = confidence
        self.horizon_days = horizon_days
        self.mc_simulations = mc_simulations

    def historical_var(
        self,
        portfolio_returns: np.ndarray,
    ) -> Tuple[float, float]:
        """Historical simulation VaR and ES.

        Args:
            portfolio_returns: Array of portfolio daily returns.

        Returns:
            (VaR, Expected Shortfall) as positive loss values.
        """
        portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
        if len(portfolio_returns) < 10:
            return 0.0, 0.0

        # Scale to horizon
        if self.horizon_days > 1:
            scaled = portfolio_returns * np.sqrt(self.horizon_days)
        else:
            scaled = portfolio_returns

        var_pct = float(np.percentile(scaled, (1 - self.confidence) * 100))
        # ES = mean of returns below VaR
        tail = scaled[scaled <= var_pct]
        es = float(np.mean(tail)) if len(tail) > 0 else var_pct

        return -var_pct, -es  # Return as positive losses

    def parametric_var(
        self,
        portfolio_returns: np.ndarray,
    ) -> float:
        """Parametric (Gaussian) VaR.

        VaR = -mu + z_alpha * sigma * sqrt(horizon)
        """
        portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
        if len(portfolio_returns) < 10:
            return 0.0

        mu = float(np.mean(portfolio_returns))
        sigma = float(np.std(portfolio_returns, ddof=1))
        z = stats.norm.ppf(self.confidence)

        var = -(mu * self.horizon_days) + z * sigma * np.sqrt(self.horizon_days)
        return max(float(var), 0.0)

    def montecarlo_var(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Monte Carlo VaR with correlated simulations.

        Args:
            returns_matrix: (T, N) matrix of asset returns.
            weights: (N,) portfolio weights.

        Returns:
            VaR as positive loss value.
        """
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        if returns_matrix.shape[0] < 10:
            return 0.0

        mu = np.mean(returns_matrix, axis=0)
        cov = np.cov(returns_matrix, rowvar=False)

        # Simulate
        try:
            simulated = np.random.multivariate_normal(
                mu * self.horizon_days,
                cov * self.horizon_days,
                size=self.mc_simulations,
            )
        except np.linalg.LinAlgError:
            # Regularize covariance
            cov += np.eye(cov.shape[0]) * 1e-6
            simulated = np.random.multivariate_normal(
                mu * self.horizon_days,
                cov * self.horizon_days,
                size=self.mc_simulations,
            )

        portfolio_sims = simulated @ weights
        var_pct = float(np.percentile(portfolio_sims, (1 - self.confidence) * 100))
        return max(-var_pct, 0.0)

    def compute(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        portfolio_value: float = 1_000_000.0,
    ) -> VaRResult:
        """Compute all three VaR methods.

        Args:
            returns_matrix: (T, N) matrix of asset returns.
            weights: (N,) portfolio weights.
            portfolio_value: Portfolio value for dollar VaR.

        Returns:
            VaRResult with all methods.
        """
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        # Portfolio returns for historical/parametric
        port_returns = returns_matrix @ weights

        var_hist, es_hist = self.historical_var(port_returns)
        var_param = self.parametric_var(port_returns)
        var_mc = self.montecarlo_var(returns_matrix, weights)

        result = VaRResult(
            var_historical=var_hist,
            var_parametric=var_param,
            var_montecarlo=var_mc,
            es_historical=es_hist,
            confidence=self.confidence,
            horizon_days=self.horizon_days,
            portfolio_value=portfolio_value,
            var_dollar_hist=var_hist * portfolio_value,
            var_dollar_param=var_param * portfolio_value,
            var_dollar_mc=var_mc * portfolio_value,
        )

        logger.info(
            "VaR (%.0f%% %d-day): Hist=%.4f, Param=%.4f, MC=%.4f",
            self.confidence * 100, self.horizon_days,
            var_hist, var_param, var_mc,
        )
        return result


# ---------------------------------------------------------------------------
# Item 17 — StressTestEngine
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """A stress test scenario definition."""
    name: str
    description: str
    shocks: Dict[str, float]  # factor/asset -> shock (e.g., {"SPX": -0.35})
    vix_shock: float = 0.0    # VIX change (e.g., +50 points)
    rate_shock: float = 0.0   # Rate change (e.g., +0.02 for 200bps)
    correlation_override: Optional[float] = None  # Force all correlations to this


@dataclass
class StressTestResult:
    """Result of a single stress test."""
    scenario_name: str = ""
    portfolio_loss: float = 0.0       # Portfolio loss (positive = loss)
    loss_pct: float = 0.0              # As percentage of portfolio
    worst_asset: str = ""
    worst_asset_loss: float = 0.0
    asset_losses: Dict[str, float] = field(default_factory=dict)
    survives: bool = True              # Loss below survival threshold


class StressTestEngine:
    """Stress testing engine with predefined and custom scenarios.

    Built-in scenarios:
      - COVID crash (March 2020): SPX -35%, VIX +50
      - Rate shock: rates +200bps, stocks -10%
      - Flash crash: SPX -10% intraday, VIX +30
      - Liquidity crisis: all correlations → 0.9, stocks -20%
    """

    # Predefined scenarios
    SCENARIOS = {
        "covid_crash": StressScenario(
            name="COVID Crash",
            description="March 2020 pandemic selloff",
            shocks={"equity": -0.35, "credit": -0.10, "commodities": -0.25},
            vix_shock=50.0,
        ),
        "rate_shock": StressScenario(
            name="Rate Shock",
            description="200bps rate rise over 1 month",
            shocks={"equity": -0.10, "bonds": -0.08, "reits": -0.15},
            rate_shock=0.02,
        ),
        "flash_crash": StressScenario(
            name="Flash Crash",
            description="Intraday flash crash",
            shocks={"equity": -0.10, "etf": -0.12},
            vix_shock=30.0,
        ),
        "liquidity_crisis": StressScenario(
            name="Liquidity Crisis",
            description="Correlation spike + liquidity withdrawal",
            shocks={"equity": -0.20, "credit": -0.15, "commodities": -0.15},
            correlation_override=0.9,
        ),
    }

    def __init__(self, survival_threshold: float = 0.20):
        """
        Args:
            survival_threshold: Maximum acceptable loss (default 20%).
        """
        self.survival_threshold = survival_threshold
        self._custom_scenarios: Dict[str, StressScenario] = {}
        self._results: List[StressTestResult] = []

    def add_scenario(self, key: str, scenario: StressScenario) -> None:
        """Add a custom stress scenario."""
        self._custom_scenarios[key] = scenario

    def run_scenario(
        self,
        scenario_key: str,
        weights: np.ndarray,
        asset_exposures: Optional[Dict[int, str]] = None,
        portfolio_value: float = 1_000_000.0,
    ) -> StressTestResult:
        """Run a single stress scenario.

        Args:
            scenario_key: Key of predefined or custom scenario.
            weights: Portfolio weights (N,).
            asset_exposures: Map of asset index → exposure type (e.g., {0: "equity"}).
            portfolio_value: Portfolio value.

        Returns:
            StressTestResult with losses.
        """
        scenario = self._custom_scenarios.get(
            scenario_key, self.SCENARIOS.get(scenario_key)
        )
        if scenario is None:
            return StressTestResult(scenario_name=scenario_key, survives=True)

        weights = np.asarray(weights, dtype=np.float64)
        n = len(weights)

        # Map assets to shocks
        default_shock = next(iter(scenario.shocks.values()), 0.0)
        asset_shocks = np.full(n, default_shock)

        if asset_exposures:
            for idx, expo_type in asset_exposures.items():
                if idx < n and expo_type in scenario.shocks:
                    asset_shocks[idx] = scenario.shocks[expo_type]

        # Portfolio loss = sum(weight_i * shock_i)
        asset_losses = weights * asset_shocks
        portfolio_loss = float(np.sum(asset_losses))

        # Find worst asset
        worst_idx = int(np.argmin(asset_losses))
        worst_name = ""
        if asset_exposures:
            worst_name = asset_exposures.get(worst_idx, f"asset_{worst_idx}")
        else:
            worst_name = f"asset_{worst_idx}"

        loss_dict = {}
        for i in range(n):
            name = asset_exposures.get(i, f"asset_{i}") if asset_exposures else f"asset_{i}"
            loss_dict[name] = float(asset_losses[i])

        result = StressTestResult(
            scenario_name=scenario.name,
            portfolio_loss=abs(portfolio_loss) * portfolio_value,
            loss_pct=abs(portfolio_loss),
            worst_asset=worst_name,
            worst_asset_loss=float(asset_losses[worst_idx]),
            asset_losses=loss_dict,
            survives=abs(portfolio_loss) < self.survival_threshold,
        )

        self._results.append(result)
        logger.info(
            "Stress test '%s': loss=%.1f%% ($%,.0f) — %s",
            scenario.name, result.loss_pct * 100, result.portfolio_loss,
            "SURVIVES" if result.survives else "FAILS",
        )
        return result

    def run_all(
        self,
        weights: np.ndarray,
        asset_exposures: Optional[Dict[int, str]] = None,
        portfolio_value: float = 1_000_000.0,
    ) -> List[StressTestResult]:
        """Run all predefined + custom scenarios.

        Returns:
            List of StressTestResult for each scenario.
        """
        results = []
        all_scenarios = {**self.SCENARIOS, **self._custom_scenarios}
        for key in all_scenarios:
            r = self.run_scenario(key, weights, asset_exposures, portfolio_value)
            results.append(r)
        return results

    @property
    def results(self) -> List[StressTestResult]:
        return self._results
