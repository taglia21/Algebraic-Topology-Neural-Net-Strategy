"""
Fama-French Factor Exposure Monitor
====================================

Tracks portfolio exposure to the Fama-French 5 factors + Momentum:
  MKT-RF   — Market excess return
  SMB      — Small Minus Big (size)
  HML      — High Minus Low (value)
  RMW      — Robust Minus Weak (profitability)
  CMA      — Conservative Minus Aggressive (investment)
  UMD      — Up Minus Down (momentum)

Uses pre-computed factor betas per stock (approximated from sector/size
characteristics when live factor returns are unavailable).

Integration:
    from src.factor_monitor import FactorMonitor

    monitor = FactorMonitor()
    exposures = monitor.get_factor_exposures(["AAPL", "JPM", "XOM"], weights)
    if not monitor.is_factor_neutral():
        print("WARNING: Factor tilt detected!")
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# FACTOR DEFINITIONS
# ============================================================================

class Factor(Enum):
    MKT_RF = "MKT-RF"
    SMB = "SMB"
    HML = "HML"
    RMW = "RMW"
    CMA = "CMA"
    UMD = "UMD"


FACTOR_NAMES = [f.value for f in Factor]


@dataclass
class FactorExposure:
    """Portfolio-level factor exposure report."""
    factor_betas: Dict[str, float]         # Factor name → portfolio beta
    factor_contributions: Dict[str, float] # Factor name → % contribution to risk
    is_neutral: bool                       # All within tolerance?
    violations: List[str]                  # Factors exceeding threshold
    gross_exposure: float                  # Sum of absolute factor betas
    net_exposure: float                    # Sum of signed factor betas
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "factor_betas": self.factor_betas,
            "factor_contributions": self.factor_contributions,
            "is_neutral": self.is_neutral,
            "violations": self.violations,
            "gross_exposure": round(self.gross_exposure, 4),
            "net_exposure": round(self.net_exposure, 4),
        }


# ============================================================================
# STOCK FACTOR BETAS (approximated from fundamental characteristics)
# ============================================================================
# These are approximate sector-level betas. In production, you'd load
# these from a factor model like Barra USE4 or do rolling regressions
# on Ken French data.

# Format: {symbol: {factor: beta}}
# Estimated from typical sector characteristics:
#   Tech → high MKT, negative HML (growth), positive RMW, positive UMD
#   Financials → high MKT, positive HML (value), positive SMB
#   Energy → moderate MKT, positive HML, positive CMA
#   Healthcare → defensive MKT, mixed factors
#   Consumer Disc → high MKT, positive UMD
#   Consumer Staples → low MKT, positive RMW, positive CMA
#   Industrials → moderate MKT, positive SMB
#   etc.

_SECTOR_FACTOR_BETAS: Dict[str, Dict[str, float]] = {
    "technology": {
        "MKT-RF": 1.15, "SMB": -0.10, "HML": -0.35,
        "RMW": 0.15, "CMA": -0.25, "UMD": 0.10,
    },
    "healthcare": {
        "MKT-RF": 0.85, "SMB": -0.05, "HML": -0.10,
        "RMW": 0.20, "CMA": 0.10, "UMD": 0.05,
    },
    "financials": {
        "MKT-RF": 1.10, "SMB": 0.05, "HML": 0.40,
        "RMW": 0.05, "CMA": 0.15, "UMD": -0.05,
    },
    "energy": {
        "MKT-RF": 1.05, "SMB": 0.10, "HML": 0.30,
        "RMW": 0.10, "CMA": 0.30, "UMD": -0.15,
    },
    "consumer_discretionary": {
        "MKT-RF": 1.10, "SMB": 0.00, "HML": -0.20,
        "RMW": 0.10, "CMA": -0.10, "UMD": 0.15,
    },
    "consumer_staples": {
        "MKT-RF": 0.65, "SMB": -0.15, "HML": 0.05,
        "RMW": 0.25, "CMA": 0.20, "UMD": 0.00,
    },
    "industrials": {
        "MKT-RF": 1.00, "SMB": 0.15, "HML": 0.10,
        "RMW": 0.10, "CMA": 0.05, "UMD": 0.05,
    },
    "utilities": {
        "MKT-RF": 0.50, "SMB": -0.10, "HML": 0.20,
        "RMW": 0.15, "CMA": 0.25, "UMD": -0.10,
    },
    "materials": {
        "MKT-RF": 1.00, "SMB": 0.20, "HML": 0.15,
        "RMW": 0.05, "CMA": 0.10, "UMD": 0.00,
    },
    "reits": {
        "MKT-RF": 0.80, "SMB": 0.10, "HML": 0.30,
        "RMW": 0.10, "CMA": 0.15, "UMD": -0.05,
    },
}

# Per-stock overrides for large-caps with well-known characteristics
_STOCK_FACTOR_OVERRIDES: Dict[str, Dict[str, float]] = {
    # Mega-cap tech (lower beta than sector avg, strong momentum)
    "AAPL": {"MKT-RF": 1.05, "SMB": -0.30, "HML": -0.25, "RMW": 0.30, "CMA": -0.20, "UMD": 0.15},
    "MSFT": {"MKT-RF": 0.95, "SMB": -0.35, "HML": -0.20, "RMW": 0.35, "CMA": -0.15, "UMD": 0.10},
    "GOOGL": {"MKT-RF": 1.05, "SMB": -0.30, "HML": -0.30, "RMW": 0.25, "CMA": -0.25, "UMD": 0.08},
    "META": {"MKT-RF": 1.20, "SMB": -0.25, "HML": -0.40, "RMW": 0.20, "CMA": -0.30, "UMD": 0.20},
    "NVDA": {"MKT-RF": 1.45, "SMB": -0.15, "HML": -0.50, "RMW": 0.10, "CMA": -0.35, "UMD": 0.30},
    "TSLA": {"MKT-RF": 1.50, "SMB": -0.10, "HML": -0.60, "RMW": -0.10, "CMA": -0.40, "UMD": 0.25},
    "AMZN": {"MKT-RF": 1.15, "SMB": -0.35, "HML": -0.45, "RMW": 0.05, "CMA": -0.30, "UMD": 0.12},

    # Large-cap financials (value tilt)
    "JPM": {"MKT-RF": 1.10, "SMB": -0.20, "HML": 0.45, "RMW": 0.15, "CMA": 0.10, "UMD": 0.05},
    "GS": {"MKT-RF": 1.25, "SMB": -0.15, "HML": 0.35, "RMW": 0.10, "CMA": 0.05, "UMD": 0.10},
    "BAC": {"MKT-RF": 1.30, "SMB": -0.10, "HML": 0.50, "RMW": 0.00, "CMA": 0.15, "UMD": -0.05},

    # Energy (value, low momentum)
    "XOM": {"MKT-RF": 0.90, "SMB": -0.20, "HML": 0.35, "RMW": 0.15, "CMA": 0.30, "UMD": -0.10},
    "CVX": {"MKT-RF": 0.95, "SMB": -0.15, "HML": 0.30, "RMW": 0.20, "CMA": 0.25, "UMD": -0.08},

    # Defensive healthcare
    "UNH": {"MKT-RF": 0.80, "SMB": -0.30, "HML": -0.05, "RMW": 0.30, "CMA": 0.05, "UMD": 0.15},
    "JNJ": {"MKT-RF": 0.65, "SMB": -0.25, "HML": 0.10, "RMW": 0.25, "CMA": 0.15, "UMD": -0.05},

    # Broad index (beta = 1 by definition)
    "SPY": {"MKT-RF": 1.00, "SMB": -0.05, "HML": 0.00, "RMW": 0.05, "CMA": 0.00, "UMD": 0.02},
    "QQQ": {"MKT-RF": 1.10, "SMB": -0.25, "HML": -0.30, "RMW": 0.15, "CMA": -0.20, "UMD": 0.10},
}

# Sector mapping for symbols not in overrides
_SYMBOL_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "GOOGL": "technology", "META": "technology", "CRM": "technology",
    "ADBE": "technology", "INTC": "technology", "AMD": "technology",
    "CSCO": "technology", "ORCL": "technology", "AVGO": "technology",
    "QCOM": "technology", "TXN": "technology",
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare",
    "ABBV": "healthcare", "PFE": "healthcare", "MRK": "healthcare",
    "TMO": "healthcare", "ABT": "healthcare",
    "JPM": "financials", "GS": "financials", "MS": "financials",
    "BAC": "financials", "WFC": "financials", "C": "financials",
    "V": "financials", "MA": "financials", "BLK": "financials",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "SLB": "energy", "EOG": "energy",
    "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
    "HD": "consumer_discretionary", "MCD": "consumer_discretionary",
    "NKE": "consumer_discretionary", "LOW": "consumer_discretionary",
    "KO": "consumer_staples", "PG": "consumer_staples",
    "PEP": "consumer_staples", "COST": "consumer_staples",
    "WMT": "consumer_staples",
    "CAT": "industrials", "HON": "industrials", "GE": "industrials",
    "DE": "industrials", "UPS": "industrials", "BA": "industrials",
    "LMT": "industrials", "RTX": "industrials",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
    "LIN": "materials", "SHW": "materials", "FCX": "materials",
    "AMT": "reits", "PLD": "reits", "CCI": "reits", "O": "reits",
    "SPY": "technology",  # Broad; use tech as proxy (high-weight)
    "QQQ": "technology",
    "IWM": "industrials",
}


def _get_stock_betas(symbol: str) -> Dict[str, float]:
    """Get factor betas for a single stock."""
    # Check per-stock overrides first
    if symbol in _STOCK_FACTOR_OVERRIDES:
        return _STOCK_FACTOR_OVERRIDES[symbol].copy()

    # Fall back to sector average
    sector = _SYMBOL_SECTOR_MAP.get(symbol, "")
    if sector in _SECTOR_FACTOR_BETAS:
        return _SECTOR_FACTOR_BETAS[sector].copy()

    # Unknown stock: assume market-neutral
    return {f: (1.0 if f == "MKT-RF" else 0.0) for f in FACTOR_NAMES}


# ============================================================================
# FACTOR MONITOR
# ============================================================================

class FactorMonitor:
    """
    Monitors portfolio factor exposures and alerts when tilts are too large.

    Usage:
        monitor = FactorMonitor()
        exposures = monitor.get_factor_exposures(["AAPL", "JPM"], {"AAPL": 0.5, "JPM": 0.5})
        if not exposures.is_neutral:
            # Alert risk manager
    """

    def __init__(
        self,
        neutral_tolerance: float = 0.20,
        mkt_tolerance: float = 0.30,
        max_single_factor: float = 0.40,
    ):
        """
        Parameters
        ----------
        neutral_tolerance : float
            Max absolute beta for non-market factors to be considered "neutral".
        mkt_tolerance : float
            Tolerance for market beta around 1.0 (i.e., 1.0 ± tolerance).
        max_single_factor : float
            Hard limit for any single factor exposure.
        """
        self.neutral_tolerance = neutral_tolerance
        self.mkt_tolerance = mkt_tolerance
        self.max_single_factor = max_single_factor
        self._history: List[FactorExposure] = []

    def get_factor_exposures(
        self,
        positions: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> FactorExposure:
        """
        Compute portfolio-level factor exposures.

        Parameters
        ----------
        positions : list of str
            Symbols currently held.
        weights : dict, optional
            {symbol: weight} where weights sum to 1.0.
            If None, assumes equal-weighted.

        Returns
        -------
        FactorExposure
            Portfolio factor betas and neutrality assessment.
        """
        if not positions:
            return FactorExposure(
                factor_betas={f: 0.0 for f in FACTOR_NAMES},
                factor_contributions={f: 0.0 for f in FACTOR_NAMES},
                is_neutral=True,
                violations=[],
                gross_exposure=0.0,
                net_exposure=0.0,
            )

        # Default to equal weight
        if weights is None:
            w = 1.0 / len(positions)
            weights = {sym: w for sym in positions}

        # Normalize weights to sum to 1
        total_w = sum(weights.get(s, 0) for s in positions)
        if total_w > 0:
            norm_weights = {s: weights.get(s, 0) / total_w for s in positions}
        else:
            norm_weights = {s: 1.0 / len(positions) for s in positions}

        # Compute portfolio-weighted factor betas
        portfolio_betas: Dict[str, float] = {f: 0.0 for f in FACTOR_NAMES}
        for sym in positions:
            w = norm_weights.get(sym, 0)
            stock_betas = _get_stock_betas(sym)
            for factor in FACTOR_NAMES:
                portfolio_betas[factor] += w * stock_betas.get(factor, 0)

        # Round
        portfolio_betas = {f: round(v, 4) for f, v in portfolio_betas.items()}

        # Compute risk contributions (approximate: proportional to beta^2)
        total_risk = sum(b ** 2 for b in portfolio_betas.values()) or 1e-10
        contributions = {
            f: round(b ** 2 / total_risk, 4)
            for f, b in portfolio_betas.items()
        }

        # Check neutrality
        violations = []
        for factor, beta in portfolio_betas.items():
            if factor == "MKT-RF":
                # Market beta: check if within [1-tol, 1+tol]
                if abs(beta - 1.0) > self.mkt_tolerance:
                    violations.append(
                        f"{factor}: {beta:.3f} (target: 1.0 ± {self.mkt_tolerance})"
                    )
            else:
                if abs(beta) > self.neutral_tolerance:
                    violations.append(
                        f"{factor}: {beta:+.3f} (limit: ±{self.neutral_tolerance})"
                    )

            # Hard limit on any single factor
            if abs(beta) > self.max_single_factor and factor != "MKT-RF":
                violations.append(
                    f"{factor}: {beta:+.3f} EXCEEDS hard limit ±{self.max_single_factor}"
                )

        is_neutral = len(violations) == 0

        gross = sum(abs(b) for f, b in portfolio_betas.items() if f != "MKT-RF")
        net = sum(b for f, b in portfolio_betas.items() if f != "MKT-RF")

        exposure = FactorExposure(
            factor_betas=portfolio_betas,
            factor_contributions=contributions,
            is_neutral=is_neutral,
            violations=violations,
            gross_exposure=round(gross, 4),
            net_exposure=round(net, 4),
        )

        self._history.append(exposure)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return exposure

    def is_factor_neutral(
        self,
        positions: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        tolerance: Optional[float] = None,
    ) -> bool:
        """
        Quick check: is the portfolio factor-neutral within tolerance?

        If positions are provided, re-computes. Otherwise uses last cached result.
        """
        if positions is not None:
            old_tol = self.neutral_tolerance
            if tolerance is not None:
                self.neutral_tolerance = tolerance
            result = self.get_factor_exposures(positions, weights)
            self.neutral_tolerance = old_tol
            return result.is_neutral

        if self._history:
            return self._history[-1].is_neutral
        return True  # No data → assume neutral

    def get_largest_tilt(self) -> Tuple[str, float]:
        """Return the factor with the largest absolute non-market tilt."""
        if not self._history:
            return "none", 0.0

        betas = self._history[-1].factor_betas
        max_factor = ""
        max_tilt = 0.0
        for f, b in betas.items():
            if f == "MKT-RF":
                continue
            if abs(b) > max_tilt:
                max_tilt = abs(b)
                max_factor = f
        return max_factor, max_tilt

    def get_hedging_suggestions(self) -> List[str]:
        """
        Suggest hedges to neutralize factor tilts.

        Returns list of human-readable suggestions.
        """
        if not self._history:
            return []

        betas = self._history[-1].factor_betas
        suggestions = []

        if betas.get("HML", 0) < -self.neutral_tolerance:
            suggestions.append(
                f"Growth tilt (HML={betas['HML']:+.3f}): "
                "Add value exposure (e.g., XLF, XLE) or short QQQ"
            )
        elif betas.get("HML", 0) > self.neutral_tolerance:
            suggestions.append(
                f"Value tilt (HML={betas['HML']:+.3f}): "
                "Add growth exposure (e.g., QQQ, XLK)"
            )

        if betas.get("SMB", 0) < -self.neutral_tolerance:
            suggestions.append(
                f"Large-cap tilt (SMB={betas['SMB']:+.3f}): "
                "Add small-cap exposure (e.g., IWM)"
            )
        elif betas.get("SMB", 0) > self.neutral_tolerance:
            suggestions.append(
                f"Small-cap tilt (SMB={betas['SMB']:+.3f}): "
                "Add large-cap exposure (e.g., SPY) or reduce small-cap"
            )

        mkt_beta = betas.get("MKT-RF", 1.0)
        if mkt_beta > 1.0 + self.mkt_tolerance:
            suggestions.append(
                f"High beta ({mkt_beta:.3f}): "
                "Reduce equity exposure or add short SPY hedge"
            )
        elif mkt_beta < 1.0 - self.mkt_tolerance:
            suggestions.append(
                f"Low beta ({mkt_beta:.3f}): "
                "Increase equity exposure or remove hedges"
            )

        if abs(betas.get("UMD", 0)) > self.neutral_tolerance:
            direction = "long momentum" if betas["UMD"] > 0 else "short momentum"
            suggestions.append(
                f"Momentum tilt ({direction}, UMD={betas['UMD']:+.3f}): "
                "Rebalance to reduce momentum concentration"
            )

        return suggestions

    def print_report(self):
        """Print a formatted factor exposure report."""
        if not self._history:
            print("No factor exposure data available.")
            return

        exp = self._history[-1]
        print("\n" + "=" * 56)
        print(f"{'FACTOR EXPOSURE REPORT':^56}")
        print("=" * 56)
        print(f"  {'Factor':<12} {'Beta':>8} {'Risk Contrib':>14} {'Status':>12}")
        print("-" * 56)

        for factor in FACTOR_NAMES:
            beta = exp.factor_betas.get(factor, 0)
            contrib = exp.factor_contributions.get(factor, 0)

            if factor == "MKT-RF":
                ok = abs(beta - 1.0) <= self.mkt_tolerance
            else:
                ok = abs(beta) <= self.neutral_tolerance

            status = "OK" if ok else "WARNING"
            print(f"  {factor:<12} {beta:>+8.3f} {contrib:>13.1%} {status:>12}")

        print("-" * 56)
        print(f"  {'Gross (ex-MKT):':<26} {exp.gross_exposure:>8.3f}")
        print(f"  {'Net (ex-MKT):':<26} {exp.net_exposure:>+8.3f}")
        print(f"  {'Neutral:':<26} {'YES' if exp.is_neutral else 'NO':>8}")

        if exp.violations:
            print(f"\n  Violations:")
            for v in exp.violations:
                print(f"    - {v}")

        suggestions = self.get_hedging_suggestions()
        if suggestions:
            print(f"\n  Hedging suggestions:")
            for s in suggestions:
                print(f"    -> {s}")

        print("=" * 56)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    monitor = FactorMonitor(neutral_tolerance=0.20)

    # Test 1: Tech-heavy portfolio (should show growth tilt)
    print("\n--- Test 1: Tech-heavy portfolio ---")
    tech_portfolio = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
    exp = monitor.get_factor_exposures(tech_portfolio)
    monitor.print_report()

    # Test 2: Balanced portfolio
    print("\n--- Test 2: Balanced portfolio ---")
    balanced = ["AAPL", "JPM", "XOM", "UNH", "CAT"]
    exp = monitor.get_factor_exposures(balanced)
    monitor.print_report()

    # Test 3: Custom weights (value tilt)
    print("\n--- Test 3: Value-tilted portfolio ---")
    value_syms = ["JPM", "BAC", "XOM", "CVX", "GS"]
    exp = monitor.get_factor_exposures(value_syms)
    monitor.print_report()

    # Test 4: is_factor_neutral
    print(f"\nTech neutral? {monitor.is_factor_neutral(tech_portfolio)}")
    print(f"Balanced neutral? {monitor.is_factor_neutral(balanced)}")

    tilt_factor, tilt_size = monitor.get_largest_tilt()
    print(f"Largest tilt: {tilt_factor} = {tilt_size:.3f}")
