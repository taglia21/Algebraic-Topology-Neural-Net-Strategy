"""
Pairs / Statistical Arbitrage Trading
=======================================
Cointegration-based stat-arb for correlated pairs.

Default pairs:
    SPY / QQQ   — broad market beta
    XLF / KRE   — large-cap vs regional banks

Entry: |z-score of spread| > 2.0
Exit:  |z-score of spread| < 0.5
Emergency exit: |z| > 4.0 (regime break)

Usage:
    from src.pairs_trading import PairsTrader, PairConfig
    pt = PairsTrader()
    signals = pt.score_all_pairs(bars_map)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────

class PairAction(str, Enum):
    LONG_A_SHORT_B = "LONG_A_SHORT_B"   # spread too low → buy A, sell B
    SHORT_A_LONG_B = "SHORT_A_LONG_B"   # spread too high → sell A, buy B
    EXIT = "EXIT"                       # close both legs
    EMERGENCY_EXIT = "EMERGENCY_EXIT"   # spread blew up
    HOLD = "HOLD"


@dataclass
class PairSignal:
    """Signal for a single pair."""
    sym_a: str
    sym_b: str
    action: str
    z_score: float
    spread_mean: float
    spread_std: float
    hedge_ratio: float
    half_life: float
    confidence: float
    reasons: List[str]


@dataclass
class PairDef:
    """Definition of a tradeable pair."""
    sym_a: str
    sym_b: str
    lookback: int = 60          # bars for hedge-ratio / z-score
    z_entry: float = 2.0        # |z| > 2 → entry
    z_exit: float = 0.5         # |z| < 0.5 → exit
    z_emergency: float = 4.0    # |z| > 4 → emergency exit


@dataclass
class PairsConfig:
    """Top-level pairs trading configuration."""
    pairs: List[PairDef] = field(default_factory=lambda: [
        PairDef("SPY", "QQQ"),
        PairDef("XLF", "KRE"),
    ])
    min_bars: int = 60
    max_half_life: float = 30.0        # reject pairs with HL > 30 days
    min_correlation: float = 0.60      # reject pairs with corr < 0.6
    position_pct: float = 0.03         # 3% equity per leg


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """OLS hedge ratio β such that spread = y - β·x is stationary."""
    if len(y) < 10 or len(x) < 10:
        return 1.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10))
    return beta


def _spread_z(y: np.ndarray, x: np.ndarray, beta: float) -> Tuple[float, float, float]:
    """Compute z-score of the spread y - β*x. Returns (z, mean, std)."""
    spread = y - beta * x
    mu = float(np.mean(spread))
    sigma = float(np.std(spread, ddof=1)) + 1e-10
    z = float((spread[-1] - mu) / sigma)
    return z, mu, sigma


def _half_life(spread: np.ndarray) -> float:
    """
    Mean-reversion half-life via OLS on Δspread ~ spread_lag.
    Returns days.  Positive = mean-reverting.
    """
    if len(spread) < 10:
        return 999.0
    lag = spread[:-1]
    delta = np.diff(spread)
    lag_mean = np.mean(lag)
    cov = float(np.sum((lag - lag_mean) * delta))
    var = float(np.sum((lag - lag_mean) ** 2)) + 1e-10
    phi = cov / var
    if phi >= 0:
        return 999.0  # diverging, not mean-reverting
    hl = -np.log(2) / phi
    return float(max(hl, 0.1))


def _adf_check(spread: np.ndarray) -> bool:
    """
    Quick Augmented Dickey-Fuller stationarity check (simplified).
    Returns True if spread is likely stationary at ~5% level.
    """
    if len(spread) < 20:
        return False
    lag = spread[:-1]
    delta = np.diff(spread)
    # OLS: delta = alpha + gamma * lag
    n = len(delta)
    x = np.column_stack([np.ones(n), lag])
    try:
        beta = np.linalg.lstsq(x, delta, rcond=None)[0]
    except np.linalg.LinAlgError:
        return False
    gamma = beta[1]
    residuals = delta - x @ beta
    se_gamma = float(np.sqrt(np.sum(residuals ** 2) / (n - 2) / (np.sum((lag - np.mean(lag)) ** 2) + 1e-10)))
    if se_gamma < 1e-12:
        return False
    t_stat = gamma / se_gamma
    # ADF 5% critical value for n~60 is roughly -2.89
    return t_stat < -2.89


# ─────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────

class PairsTrader:
    """
    Cointegration-based statistical arbitrage engine.

    For each configured pair (A, B):
      1. Compute OLS hedge ratio β
      2. Build spread = close_A − β · close_B
      3. Compute z-score of the spread
      4. Enter when |z| > z_entry, exit when |z| < z_exit
    """

    def __init__(self, config: PairsConfig = None):
        self.cfg = config or PairsConfig()
        # Track which pairs are currently "active" (have an open position)
        self._active_pairs: Dict[str, str] = {}  # "A|B" → action

    def score_pair(
        self,
        pair: PairDef,
        bars_a: List[dict],
        bars_b: List[dict],
    ) -> PairSignal:
        """
        Score a single pair and produce an action signal.

        Parameters
        ----------
        pair : PairDef
        bars_a, bars_b : list[dict]
            Alpaca-style OHLCV bars for each leg.

        Returns
        -------
        PairSignal
        """
        hold = PairSignal(
            sym_a=pair.sym_a, sym_b=pair.sym_b, action="HOLD",
            z_score=0.0, spread_mean=0.0, spread_std=0.0,
            hedge_ratio=1.0, half_life=999.0, confidence=0.0,
            reasons=["insufficient data"],
        )

        n = min(len(bars_a), len(bars_b))
        if n < self.cfg.min_bars:
            return hold

        closes_a = np.array([float(b["c"]) for b in bars_a[-n:]])
        closes_b = np.array([float(b["c"]) for b in bars_b[-n:]])

        # Correlation check
        corr = float(np.corrcoef(closes_a, closes_b)[0, 1])
        if abs(corr) < self.cfg.min_correlation:
            hold.reasons = [f"corr={corr:.2f} < {self.cfg.min_correlation}"]
            return hold

        # Hedge ratio & spread
        beta = _ols_hedge_ratio(closes_a, closes_b)
        spread = closes_a - beta * closes_b
        z, mu, sigma = _spread_z(closes_a, closes_b, beta)
        hl = _half_life(spread)

        # Stationarity check
        is_stationary = _adf_check(spread)

        pair_key = f"{pair.sym_a}|{pair.sym_b}"
        currently_active = pair_key in self._active_pairs
        reasons: List[str] = []
        action = "HOLD"
        confidence = 0.0

        # ── Emergency exit ────────────────────────────────────────
        if currently_active and abs(z) > pair.z_emergency:
            action = "EMERGENCY_EXIT"
            confidence = 0.95
            reasons.append(f"|z|={abs(z):.2f}>{pair.z_emergency} EMERGENCY")
            self._active_pairs.pop(pair_key, None)

        # ── Normal exit ───────────────────────────────────────────
        elif currently_active and abs(z) < pair.z_exit:
            action = "EXIT"
            confidence = 0.7
            reasons.append(f"|z|={abs(z):.2f}<{pair.z_exit} → close pair")
            self._active_pairs.pop(pair_key, None)

        # ── Entry: spread too low → long A, short B ──────────────
        elif not currently_active and z < -pair.z_entry:
            if hl <= self.cfg.max_half_life and is_stationary:
                action = "LONG_A_SHORT_B"
                confidence = min(0.5 + 0.1 * (abs(z) - pair.z_entry), 0.90)
                reasons.append(f"z={z:+.2f}<-{pair.z_entry}")
                reasons.append(f"HL={hl:.1f}d")
                if is_stationary:
                    reasons.append("ADF stationary")
                self._active_pairs[pair_key] = action
            else:
                reasons.append(f"HL={hl:.1f}d>{self.cfg.max_half_life} or non-stationary")

        # ── Entry: spread too high → short A, long B ─────────────
        elif not currently_active and z > pair.z_entry:
            if hl <= self.cfg.max_half_life and is_stationary:
                action = "SHORT_A_LONG_B"
                confidence = min(0.5 + 0.1 * (abs(z) - pair.z_entry), 0.90)
                reasons.append(f"z={z:+.2f}>+{pair.z_entry}")
                reasons.append(f"HL={hl:.1f}d")
                if is_stationary:
                    reasons.append("ADF stationary")
                self._active_pairs[pair_key] = action
            else:
                reasons.append(f"HL={hl:.1f}d>{self.cfg.max_half_life} or non-stationary")

        else:
            reasons.append(f"z={z:+.2f} inside bands, corr={corr:.2f}")

        return PairSignal(
            sym_a=pair.sym_a, sym_b=pair.sym_b, action=action,
            z_score=z, spread_mean=mu, spread_std=sigma,
            hedge_ratio=beta, half_life=hl, confidence=confidence,
            reasons=reasons,
        )

    def score_all_pairs(
        self, bars_map: Dict[str, List[dict]],
    ) -> List[PairSignal]:
        """Score every configured pair. Returns only actionable signals."""
        results: List[PairSignal] = []
        for pair in self.cfg.pairs:
            bars_a = bars_map.get(pair.sym_a)
            bars_b = bars_map.get(pair.sym_b)
            if bars_a is None or bars_b is None:
                continue
            sig = self.score_pair(pair, bars_a, bars_b)
            if sig.action != "HOLD":
                results.append(sig)
        return results

    @property
    def active_pairs(self) -> Dict[str, str]:
        return dict(self._active_pairs)
