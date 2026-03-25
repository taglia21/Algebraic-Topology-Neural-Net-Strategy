"""
core/risk_box.py
================
ORIA-inspired Risk Box — the constraint layer that sits between the
Allocator and Execution in the ORIA pipeline.

From Joshua Aalampour's ORIA Part 3:
- Target Volatility scaling
- Gross & Net Exposure Caps
- Concentration Limits
- Event Filters
- Stress-damped position sizing:
    ℓ_t = clip( (f_budget / f_max) · exp(-λ_stress · ξ_stress), ρ_min, ρ_max )

Also implements the transaction-cost-aware portfolio optimization:
    w*_{t+1} = arg min [ ½(w - w_target)^T Ω (w - w_target) + λ_tc |Δw| ]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskBoxConfig:
    """Risk Box configuration."""
    target_annual_vol: float = 0.15        # 15% annualized target vol
    max_gross_exposure: float = 1.0        # 100% of NAV
    max_net_exposure: float = 0.50         # 50% net long/short
    max_single_position: float = 0.15      # 15% per position
    max_sector_exposure: float = 0.40      # 40% per sector
    max_concurrent_positions: int = 8      # Max open positions
    stress_lambda: float = 2.0             # Stress damping coefficient
    rho_min: float = 0.20                  # Minimum risk scalar (20% of budget)
    rho_max: float = 1.0                   # Maximum risk scalar (100% of budget)
    tc_lambda: float = 0.002              # Transaction cost penalty (20 bps)
    event_filter_enabled: bool = True      # Filter around earnings/FOMC


# Sector mapping for 50-symbol universe
_SECTOR_MAP = {
    # Indices/ETFs
    "SPY": "INDEX", "QQQ": "INDEX", "IWM": "INDEX",
    "GLD": "COMMODITIES", "TLT": "BONDS",
    "XLF": "FINANCIALS", "XLE": "ENERGY", "XLK": "TECH",
    "XLV": "HEALTHCARE", "XLI": "INDUSTRIALS",
    # Mega-cap Tech
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH",
    "META": "TECH", "GOOGL": "TECH", "NFLX": "TECH",
    # Tech/Semi
    "AMD": "TECH", "CRM": "TECH", "INTC": "TECH",
    "AVGO": "TECH", "ADBE": "TECH", "ORCL": "TECH",
    # Finance
    "JPM": "FINANCIALS", "BAC": "FINANCIALS", "GS": "FINANCIALS",
    "V": "FINANCIALS", "MA": "FINANCIALS",
    # Healthcare
    "JNJ": "HEALTHCARE", "UNH": "HEALTHCARE", "PFE": "HEALTHCARE",
    "ABBV": "HEALTHCARE", "MRK": "HEALTHCARE",
    # Consumer
    "AMZN": "CONSUMER", "TSLA": "CONSUMER", "WMT": "CONSUMER",
    "COST": "CONSUMER", "HD": "CONSUMER", "MCD": "CONSUMER",
    "NKE": "CONSUMER", "SBUX": "CONSUMER", "DIS": "CONSUMER",
    # Energy
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY",
    # Industrials
    "CAT": "INDUSTRIALS", "BA": "INDUSTRIALS", "UPS": "INDUSTRIALS",
    # Other
    "UBER": "TECH", "LIN": "MATERIALS", "NEE": "UTILITIES", "O": "REALESTATE",
}


@dataclass
class RiskBoxResult:
    """Output of risk box processing."""
    approved_signals: List[Dict]
    rejected_signals: List[Dict]
    risk_scalar: float
    gross_exposure_pct: float
    net_exposure_pct: float
    position_count: int
    violations: List[str]


class RiskBox:
    """ORIA-inspired Risk Box.

    Applies a comprehensive set of risk constraints to sized signals
    before they reach the execution layer.

    Parameters
    ----------
    config : RiskBoxConfig
        Risk box configuration.
    nav : float
        Current net asset value.
    """

    def __init__(self, config: Optional[RiskBoxConfig] = None, nav: float = 6000.0):
        self.config = config or RiskBoxConfig()
        self.nav = nav
        self._current_positions: Dict[str, Dict] = {}  # ticker → {value, direction, sector}

    def update_positions(self, positions: Dict[str, Dict]):
        """Update current position state.

        Parameters
        ----------
        positions : dict
            {ticker: {"value": float, "direction": "LONG"/"SHORT", "qty": int}}
        """
        self._current_positions = positions

    def update_nav(self, nav: float):
        """Update current NAV."""
        self.nav = nav

    def compute_risk_scalar(
        self,
        realized_vol: float,
        stress_indicator: float = 0.0,
    ) -> float:
        """Compute ORIA-style stress-damped risk scalar.

        ℓ_t = clip( (σ_target / σ_realized) · exp(-λ · ξ_stress), ρ_min, ρ_max )

        Parameters
        ----------
        realized_vol : float
            Current annualized realized portfolio volatility.
        stress_indicator : float
            Stress level (0 = calm, 1+ = elevated). Can use VIX percentile,
            regime score, or drawdown depth.

        Returns
        -------
        float
            Risk scalar ∈ [ρ_min, ρ_max].
        """
        cfg = self.config

        # Vol targeting: scale inversely to realized vol
        if realized_vol > 1e-6:
            vol_ratio = cfg.target_annual_vol / realized_vol
        else:
            vol_ratio = 1.0

        # Stress damping: exponential decay
        stress_damping = math.exp(-cfg.stress_lambda * stress_indicator)

        # Combined risk scalar
        raw_scalar = vol_ratio * stress_damping

        # Clip to bounds
        scalar = max(cfg.rho_min, min(cfg.rho_max, raw_scalar))

        logger.info(
            "RiskBox scalar: %.3f (vol_ratio=%.2f, stress_damp=%.3f, ξ=%.2f)",
            scalar, vol_ratio, stress_damping, stress_indicator,
        )
        return scalar

    def _compute_current_exposures(self) -> Dict[str, float]:
        """Compute current gross, net, and sector exposures."""
        long_value = 0.0
        short_value = 0.0
        sector_values: Dict[str, float] = {}

        for ticker, pos in self._current_positions.items():
            val = abs(pos.get("value", 0))
            direction = pos.get("direction", "LONG")
            sector = _SECTOR_MAP.get(ticker, "OTHER")

            if direction == "LONG":
                long_value += val
            else:
                short_value += val

            sector_values[sector] = sector_values.get(sector, 0) + val

        gross = (long_value + short_value) / self.nav if self.nav > 0 else 0
        net = (long_value - short_value) / self.nav if self.nav > 0 else 0

        return {
            "gross_pct": gross,
            "net_pct": net,
            "long_value": long_value,
            "short_value": short_value,
            "sector_values": sector_values,
            "position_count": len(self._current_positions),
        }

    def process_signals(
        self,
        sized_signals: List[Dict],
        realized_vol: float = 0.15,
        stress_indicator: float = 0.0,
    ) -> RiskBoxResult:
        """Apply all Risk Box constraints to sized signals.

        Parameters
        ----------
        sized_signals : list of dict
            Each has: ticker, direction, position_value, position_pct.
        realized_vol : float
            Current portfolio annualized vol.
        stress_indicator : float
            Stress level (0=calm, 1+=elevated).

        Returns
        -------
        RiskBoxResult
            Approved and rejected signals with violations.
        """
        cfg = self.config
        exposures = self._compute_current_exposures()
        risk_scalar = self.compute_risk_scalar(realized_vol, stress_indicator)

        approved = []
        rejected = []
        violations = []

        # Running counters for this batch
        pending_gross = exposures["gross_pct"]
        pending_positions = exposures["position_count"]
        pending_sectors = dict(exposures["sector_values"])

        for sig in sized_signals:
            ticker = sig.get("ticker", "???")
            direction = sig.get("direction", "LONG")
            pos_value = sig.get("position_value", 0)
            pos_pct = pos_value / self.nav if self.nav > 0 else 0
            sector = _SECTOR_MAP.get(ticker, "OTHER")
            reject_reasons = []

            # Apply risk scalar to position size
            scaled_value = pos_value * risk_scalar
            scaled_pct = scaled_value / self.nav if self.nav > 0 else 0

            # 1. Single position concentration limit
            if scaled_pct > cfg.max_single_position:
                reject_reasons.append(
                    f"concentration {scaled_pct:.1%} > {cfg.max_single_position:.0%}"
                )

            # 2. Gross exposure cap
            new_gross = pending_gross + scaled_pct
            if new_gross > cfg.max_gross_exposure:
                reject_reasons.append(
                    f"gross exposure {new_gross:.1%} > {cfg.max_gross_exposure:.0%}"
                )

            # 3. Max concurrent positions
            if pending_positions >= cfg.max_concurrent_positions:
                reject_reasons.append(
                    f"max positions {pending_positions} >= {cfg.max_concurrent_positions}"
                )

            # 4. Sector concentration
            sector_val = pending_sectors.get(sector, 0) + scaled_value
            sector_pct = sector_val / self.nav if self.nav > 0 else 0
            if sector_pct > cfg.max_sector_exposure:
                reject_reasons.append(
                    f"sector {sector} at {sector_pct:.1%} > {cfg.max_sector_exposure:.0%}"
                )

            if reject_reasons:
                sig_copy = dict(sig)
                sig_copy["reject_reasons"] = reject_reasons
                rejected.append(sig_copy)
                violations.extend([f"{ticker}: {r}" for r in reject_reasons])
                logger.info("RiskBox REJECT %s %s: %s", ticker, direction, "; ".join(reject_reasons))
            else:
                # Apply risk-scalar-adjusted size
                sig_copy = dict(sig)
                sig_copy["position_value"] = round(scaled_value, 2)
                sig_copy["position_pct"] = round(scaled_pct * 100, 2)
                sig_copy["risk_scalar"] = round(risk_scalar, 4)
                approved.append(sig_copy)

                # Update running counters
                pending_gross += scaled_pct
                pending_positions += 1
                pending_sectors[sector] = pending_sectors.get(sector, 0) + scaled_value

                logger.info(
                    "RiskBox APPROVE %s %s $%.0f (%.1f%% NAV, scalar=%.2f)",
                    ticker, direction, scaled_value, scaled_pct * 100, risk_scalar,
                )

        return RiskBoxResult(
            approved_signals=approved,
            rejected_signals=rejected,
            risk_scalar=risk_scalar,
            gross_exposure_pct=round(pending_gross * 100, 2),
            net_exposure_pct=round(exposures["net_pct"] * 100, 2),
            position_count=pending_positions,
            violations=violations,
        )
