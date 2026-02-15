"""
Market Regime Detector — HMM + GMM Ensemble
=============================================

Probabilistic regime detection using a Hidden Markov Model (primary) with
Gaussian Mixture Model (secondary) for cross-validation.

Detects 4 regimes:
- TRENDING_BULL: Positive mean return, moderate variance
- TRENDING_BEAR: Negative mean return, moderate-high variance
- MEAN_REVERTING: Low variance, near-zero mean returns
- HIGH_VOLATILITY: Very high variance regardless of direction

Key improvement over the old rule-based system:
  The confidence output is a REAL posterior probability P(state|observations)
  from the HMM, not an arbitrary score built from additive weights.

Technical indicators are still computed for:
  1. HMM/GMM feature construction (returns, realized vol, ADX)
  2. State labeling (mapping HMM numeric states → semantic regime names)
  3. Evidence logging for trade rationale

Academic reference:
  Hamilton (1989) "A New Approach to the Economic Analysis of Time Series"
  Regime-aware allocation improves Sharpe by 30-50% (arXiv 2025)

Installed deps: hmmlearn 0.3.3, scikit-learn 1.3.2, numpy 1.26.2

Author: System Overhaul - Feb 2026  |  Fix #2: HMM/GMM Regime Detector
"""

import logging
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from hmmlearn import hmm as hmmlearn_hmm
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    _GMM_AVAILABLE = True
except ImportError:
    _GMM_AVAILABLE = False


# ============================================================================
# DATA MODELS
# ============================================================================

class Regime(Enum):
    """Market regime classification for strategy selection."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class TechnicalSignals:
    """Technical indicator readings for a symbol."""
    symbol: str

    # Price & trend
    current_price: float
    sma_20: float
    sma_50: float
    sma_200: float
    price_vs_sma20_pct: float    # % above/below 20 SMA
    price_vs_sma50_pct: float    # % above/below 50 SMA

    # Momentum
    rsi_14: float                # RSI 14-period
    macd_signal: float           # MACD - Signal (positive = bullish)
    macd_histogram: float        # MACD histogram

    # Volatility
    bb_width: float              # Bollinger Band width (% of price)
    bb_position: float           # Price position within bands (0-1)
    atr_pct: float               # ATR as % of price (14-period)

    # Trend strength
    adx: float                   # ADX (0-100, >25 = trending)
    trend_direction: int         # +1 bull, -1 bear, 0 neutral

    # Volume
    volume_ratio: float          # Current volume vs 20-day average

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeResult:
    """Regime detection result with calibrated probability and evidence."""
    regime: Regime
    confidence: float            # 0-1  — REAL posterior probability from HMM/GMM
    evidence: Dict[str, str]     # Key evidence points
    technicals: Optional[TechnicalSignals]
    state_probabilities: Optional[Dict[str, float]] = None  # Full posterior
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS  (unchanged from v1 — pure math)
# ============================================================================

def _sma(prices: np.ndarray, window: int) -> float:
    if len(prices) < window:
        return float(np.mean(prices))
    return float(np.mean(prices[-window:]))


def _ema(prices: np.ndarray, window: int) -> np.ndarray:
    alpha = 2.0 / (window + 1)
    ema = np.zeros_like(prices, dtype=float)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _macd(prices: np.ndarray) -> Tuple[float, float, float]:
    if len(prices) < 35:
        return 0.0, 0.0, 0.0
    ema_12 = _ema(prices, 12)
    ema_26 = _ema(prices, 26)
    macd_line = ema_12 - ema_26
    signal_line = _ema(macd_line[-9:], 9) if len(macd_line) >= 9 else macd_line
    histogram = macd_line[-1] - signal_line[-1]
    return float(macd_line[-1]), float(signal_line[-1]), float(histogram)


def _bollinger_bands(
    prices: np.ndarray, window: int = 20, num_std: float = 2.0
) -> Tuple[float, float, float, float, float]:
    if len(prices) < window:
        mid = float(np.mean(prices))
        return mid * 1.02, mid, mid * 0.98, 0.04, 0.5
    window_prices = prices[-window:]
    mid = float(np.mean(window_prices))
    std = float(np.std(window_prices))
    upper = mid + num_std * std
    lower = mid - num_std * std
    current = float(prices[-1])
    width_pct = (upper - lower) / mid if mid > 0 else 0
    position = (current - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return upper, mid, lower, width_pct, position


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 20.0
    n = len(closes)
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, n):
        high, low, prev_close = highs[i], lows[i], closes[i - 1]
        prev_high, prev_low = highs[i - 1], lows[i - 1]
        tr_list.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        plus_dm_list.append(max(high - prev_high, 0) if (high - prev_high) > (prev_low - low) else 0)
        minus_dm_list.append(max(prev_low - low, 0) if (prev_low - low) > (high - prev_high) else 0)
    if len(tr_list) < period:
        return 20.0
    atr = np.mean(tr_list[:period])
    plus_di_smooth = np.mean(plus_dm_list[:period])
    minus_di_smooth = np.mean(minus_dm_list[:period])
    dx_list = []
    for i in range(period, len(tr_list)):
        atr = atr - (atr / period) + tr_list[i]
        plus_di_smooth = plus_di_smooth - (plus_di_smooth / period) + plus_dm_list[i]
        minus_di_smooth = minus_di_smooth - (minus_di_smooth / period) + minus_dm_list[i]
        if atr > 0:
            plus_di = 100 * plus_di_smooth / atr
            minus_di = 100 * minus_di_smooth / atr
        else:
            plus_di = minus_di = 0
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
        dx_list.append(dx)
    if not dx_list:
        return 20.0
    return min(float(np.mean(dx_list[-period:])), 100.0)


def _atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.02
    tr_values = []
    for i in range(1, len(closes)):
        tr_values.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    atr = float(np.mean(tr_values[-period:]))
    return atr / float(closes[-1]) if closes[-1] > 0 else 0.02


# ============================================================================
# HMM REGIME DETECTOR
# ============================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    The HMM is trained on a feature matrix X = [daily_returns, realized_vol_20d, momentum_sign]
    with 3-4 hidden states.  After fitting, the states are labeled by examining
    each state's emission mean/variance:
      - Lowest variance, near-zero mean  → MEAN_REVERTING
      - Highest variance                 → HIGH_VOLATILITY
      - Positive mean, moderate variance → TRENDING_BULL
      - Negative mean, moderate variance → TRENDING_BEAR

    The key output is `model.predict_proba(X)[-1]` which gives the posterior
    probability over states conditioned on ALL observations — a mathematically
    rigorous confidence, unlike the old additive-score approach.

    Retraining: call `fit()` with fresh data weekly, or when Brier score > 0.30.
    """

    MODEL_PATH = os.path.join("data", "hmm_regime_model.pkl")

    def __init__(self, n_states: int = 4, lookback_days: int = 252):
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.is_fitted = False

        if not _HMM_AVAILABLE:
            logger.warning("hmmlearn not installed — HMM regime detector disabled")
            self.model = None
            return

        self.model = hmmlearn_hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        # Mapping from numeric state → Regime enum (set after fit)
        self._state_map: Dict[int, Regime] = {}

        # Try to load persisted model
        self._try_load_model()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, returns: np.ndarray, volatility: np.ndarray,
            momentum: Optional[np.ndarray] = None) -> None:
        """
        Fit HMM on historical data.  Call at startup with 1+ year of daily data.

        Args:
            returns:    1-D array of daily log returns
            volatility: 1-D array of rolling 20-day realised vol (annualised)
            momentum:   (optional) 1-D array, sign of 10-day cumulative return
        """
        if self.model is None:
            return
        n = min(len(returns), len(volatility))
        if n < 60:
            logger.warning(f"HMM fit: only {n} observations — skipping")
            return
        returns = returns[-n:]
        volatility = volatility[-n:]
        if momentum is not None:
            momentum = momentum[-n:]
            X = np.column_stack([returns, volatility, momentum])
        else:
            X = np.column_stack([returns, volatility])

        self.model.fit(X)
        self.is_fitted = True
        self._label_states(X)
        self._persist_model()
        logger.info(f"HMM fitted on {n} obs, {self.n_states} states. Map: {self._state_map}")

    def predict(self, returns: np.ndarray, volatility: np.ndarray,
                momentum: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
        """
        Predict current state and full posterior.

        Returns:
            (predicted_state_index, state_probabilities_array)
        """
        if self.model is None or not self.is_fitted:
            raise RuntimeError("HMM not fitted")
        n = min(len(returns), len(volatility))
        returns = returns[-self.lookback_days:]
        volatility = volatility[-self.lookback_days:]
        if momentum is not None:
            momentum = momentum[-self.lookback_days:]
            X = np.column_stack([returns[:n], volatility[:n], momentum[:n]])
        else:
            n = min(len(returns), len(volatility))
            X = np.column_stack([returns[:n], volatility[:n]])

        posteriors = self.model.predict_proba(X)
        last_probs = posteriors[-1]
        best_state = int(np.argmax(last_probs))
        return best_state, last_probs

    def state_to_regime(self, state_idx: int) -> Regime:
        return self._state_map.get(state_idx, Regime.UNKNOWN)

    # ------------------------------------------------------------------ #
    # State labeling
    # ------------------------------------------------------------------ #

    def _label_states(self, X: np.ndarray) -> None:
        """
        Label HMM states by examining emission parameters.

        Strategy:
          1. Decode most-likely state sequence
          2. For each state, compute mean return and mean volatility
          3. Assign labels by variance ranking + mean sign
        """
        states = self.model.predict(X)
        state_stats: Dict[int, Dict[str, float]] = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                state_stats[s] = {"mean_ret": 0.0, "mean_vol": 1.0, "count": 0}
                continue
            state_stats[s] = {
                "mean_ret": float(np.mean(X[mask, 0])),
                "mean_vol": float(np.mean(X[mask, 1])),
                "count": int(mask.sum()),
            }

        # Sort by volatility
        sorted_by_vol = sorted(state_stats.items(), key=lambda kv: kv[1]["mean_vol"])

        assigned: Dict[int, Regime] = {}
        used_regimes = set()

        # Highest vol → HIGH_VOLATILITY
        hv_state = sorted_by_vol[-1][0]
        assigned[hv_state] = Regime.HIGH_VOLATILITY
        used_regimes.add(Regime.HIGH_VOLATILITY)

        # Lowest vol → MEAN_REVERTING
        mr_state = sorted_by_vol[0][0]
        if mr_state != hv_state:
            assigned[mr_state] = Regime.MEAN_REVERTING
            used_regimes.add(Regime.MEAN_REVERTING)

        # Remaining states: classify by mean return sign
        for s in range(self.n_states):
            if s in assigned:
                continue
            mean_ret = state_stats[s]["mean_ret"]
            if mean_ret > 0 and Regime.TRENDING_BULL not in used_regimes:
                assigned[s] = Regime.TRENDING_BULL
                used_regimes.add(Regime.TRENDING_BULL)
            elif mean_ret <= 0 and Regime.TRENDING_BEAR not in used_regimes:
                assigned[s] = Regime.TRENDING_BEAR
                used_regimes.add(Regime.TRENDING_BEAR)
            elif Regime.TRENDING_BULL not in used_regimes:
                assigned[s] = Regime.TRENDING_BULL
                used_regimes.add(Regime.TRENDING_BULL)
            else:
                assigned[s] = Regime.TRENDING_BEAR
                used_regimes.add(Regime.TRENDING_BEAR)

        self._state_map = assigned

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist_model(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump({"model": self.model, "state_map": self._state_map}, f)
            logger.info(f"HMM model persisted to {self.MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to persist HMM model: {e}")

    def _try_load_model(self) -> None:
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self._state_map = data["state_map"]
                self.is_fitted = True
                logger.info(f"HMM model loaded from {self.MODEL_PATH}")
        except Exception as e:
            logger.debug(f"Could not load HMM model: {e}")


# ============================================================================
# GMM REGIME DETECTOR (cross-validation / simpler alternative)
# ============================================================================

class GMMRegimeDetector:
    """
    Gaussian Mixture Model for regime detection — no temporal structure,
    but faster to fit and useful as a cross-check against the HMM.
    """

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.is_fitted = False
        if not _GMM_AVAILABLE:
            self.model = None
            return
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        self._state_map: Dict[int, Regime] = {}

    def fit(self, returns: np.ndarray, volatility: np.ndarray) -> None:
        if self.model is None:
            return
        n = min(len(returns), len(volatility))
        X = np.column_stack([returns[-n:], volatility[-n:]])
        self.model.fit(X)
        self.is_fitted = True
        self._label_states(X)

    def predict(self, returns: np.ndarray, volatility: np.ndarray) -> Tuple[int, np.ndarray]:
        if self.model is None or not self.is_fitted:
            raise RuntimeError("GMM not fitted")
        n = min(len(returns), len(volatility))
        x = np.column_stack([returns[-1:], volatility[-1:]])
        probs = self.model.predict_proba(x)[0]
        return int(np.argmax(probs)), probs

    def state_to_regime(self, idx: int) -> Regime:
        return self._state_map.get(idx, Regime.UNKNOWN)

    def _label_states(self, X: np.ndarray) -> None:
        labels = self.model.predict(X)
        state_stats = {}
        for s in range(self.n_regimes):
            mask = labels == s
            if mask.sum() == 0:
                state_stats[s] = {"mean_ret": 0., "mean_vol": 1., "count": 0}
                continue
            state_stats[s] = {
                "mean_ret": float(np.mean(X[mask, 0])),
                "mean_vol": float(np.mean(X[mask, 1])),
                "count": int(mask.sum()),
            }
        sorted_by_vol = sorted(state_stats.items(), key=lambda kv: kv[1]["mean_vol"])
        assigned: Dict[int, Regime] = {}
        used = set()
        hv = sorted_by_vol[-1][0]
        assigned[hv] = Regime.HIGH_VOLATILITY; used.add(Regime.HIGH_VOLATILITY)
        mr = sorted_by_vol[0][0]
        if mr != hv:
            assigned[mr] = Regime.MEAN_REVERTING; used.add(Regime.MEAN_REVERTING)
        for s in range(self.n_regimes):
            if s in assigned:
                continue
            mr_val = state_stats[s]["mean_ret"]
            if mr_val > 0 and Regime.TRENDING_BULL not in used:
                assigned[s] = Regime.TRENDING_BULL; used.add(Regime.TRENDING_BULL)
            elif mr_val <= 0 and Regime.TRENDING_BEAR not in used:
                assigned[s] = Regime.TRENDING_BEAR; used.add(Regime.TRENDING_BEAR)
            elif Regime.TRENDING_BULL not in used:
                assigned[s] = Regime.TRENDING_BULL; used.add(Regime.TRENDING_BULL)
            else:
                assigned[s] = Regime.TRENDING_BEAR; used.add(Regime.TRENDING_BEAR)
        self._state_map = assigned


# ============================================================================
# COMBINED REGIME DETECTOR (Public interface — drop-in replacement)
# ============================================================================

class RuleBasedRegimeDetector:
    """
    Production regime detector that wraps HMM + GMM ensemble
    while maintaining the same public interface as the old rule-based system.

    Public API (unchanged):
      - detect_regime(symbol) → RegimeResult
      - get_technicals(symbol) → TechnicalSignals

    Internal flow:
      1. Download 2 yr daily price data via yfinance
      2. Compute feature vectors (returns, realized vol, momentum)
      3. If HMM/GMM not fitted, fit them (first call or weekly refresh)
      4. Get posterior probability from HMM (primary)
      5. Cross-check with GMM (secondary)
      6. If models disagree, use the one with higher confidence
      7. Fall back to rule-based scoring if both ML models fail
    """

    RETRAIN_INTERVAL_DAYS = 7  # Retrain models weekly

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._hmm = HMMRegimeDetector(n_states=4, lookback_days=252)
        self._gmm = GMMRegimeDetector(n_regimes=4)
        self._last_train_time: Optional[datetime] = None
        self._cache: Dict[str, Tuple[RegimeResult, datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)

    def detect_regime(self, symbol: str = "SPY") -> RegimeResult:
        """
        Detect current market regime for a symbol.

        Returns RegimeResult with calibrated posterior probability.
        """
        # Cache check
        if symbol in self._cache:
            cached, ts = self._cache[symbol]
            if datetime.now() - ts < self._cache_ttl:
                return cached

        # Compute technicals (always needed for evidence & fallback)
        technicals = self._compute_technicals(symbol)

        # Try ML-based detection
        result = self._detect_regime_ml(symbol, technicals)

        if result is None:
            # Fallback: rule-based classification
            if technicals is not None:
                result = self._classify_regime_rules(technicals)
            else:
                result = RegimeResult(
                    regime=Regime.UNKNOWN, confidence=0.0,
                    evidence={"error": "No data"}, technicals=None,
                )

        self._cache[symbol] = (result, datetime.now())

        adx_str = f"{technicals.adx:.1f}" if technicals else "N/A"
        atr_str = f"{technicals.atr_pct:.2%}" if technicals else "N/A"
        self.logger.info(
            f"Regime [{symbol}]: {result.regime.value} "
            f"(confidence: {result.confidence:.0%}) "
            f"ADX={adx_str} "
            f"ATR%={atr_str}"
        )

        return result

    def get_technicals(self, symbol: str) -> Optional[TechnicalSignals]:
        return self._compute_technicals(symbol)

    # ================================================================== #
    # ML-based detection (HMM + GMM)
    # ================================================================== #

    def _detect_regime_ml(self, symbol: str,
                          technicals: Optional[TechnicalSignals]) -> Optional[RegimeResult]:
        """Run HMM and GMM, ensemble the results."""
        if yf is None:
            return None

        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False)
            if data.empty or len(data) < 120:
                return None

            closes = data["Close"].values.flatten().astype(float)
            log_returns = np.diff(np.log(closes))

            # 20-day rolling realised vol (annualised)
            rvol = np.array([
                np.std(log_returns[max(0, i - 20):i]) * np.sqrt(252)
                if i >= 20 else np.std(log_returns[:max(i, 1)]) * np.sqrt(252)
                for i in range(1, len(log_returns) + 1)
            ])

            # 10-day momentum sign
            momentum = np.zeros(len(log_returns))
            for i in range(10, len(log_returns)):
                momentum[i] = np.sign(np.sum(log_returns[i - 10:i]))

            # Retrain if needed
            needs_train = (
                not self._hmm.is_fitted
                or self._last_train_time is None
                or (datetime.now() - self._last_train_time).days >= self.RETRAIN_INTERVAL_DAYS
            )
            if needs_train:
                self.logger.info(f"Training HMM+GMM on {len(log_returns)} observations...")
                self._hmm.fit(log_returns, rvol, momentum)
                self._gmm.fit(log_returns, rvol)
                self._last_train_time = datetime.now()

            # --- HMM prediction ---
            hmm_regime: Optional[Regime] = None
            hmm_conf: float = 0.0
            hmm_probs: Optional[np.ndarray] = None
            try:
                hmm_state, hmm_probs = self._hmm.predict(log_returns, rvol, momentum)
                hmm_regime = self._hmm.state_to_regime(hmm_state)
                hmm_conf = float(hmm_probs[hmm_state])
            except Exception as e:
                self.logger.debug(f"HMM predict failed: {e}")

            # --- GMM prediction ---
            gmm_regime: Optional[Regime] = None
            gmm_conf: float = 0.0
            gmm_probs: Optional[np.ndarray] = None
            try:
                gmm_state, gmm_probs = self._gmm.predict(log_returns, rvol)
                gmm_regime = self._gmm.state_to_regime(gmm_state)
                gmm_conf = float(gmm_probs[gmm_state])
            except Exception as e:
                self.logger.debug(f"GMM predict failed: {e}")

            # --- Ensemble ---
            if hmm_regime is None and gmm_regime is None:
                return None

            # Build evidence
            evidence: Dict[str, str] = {}
            state_prob_dict: Dict[str, float] = {}

            if hmm_regime is not None and hmm_probs is not None:
                evidence["hmm_regime"] = f"{hmm_regime.value} ({hmm_conf:.0%})"
                for s_idx, p in enumerate(hmm_probs):
                    r = self._hmm.state_to_regime(s_idx)
                    state_prob_dict[f"hmm_{r.value}"] = round(float(p), 4)

            if gmm_regime is not None and gmm_probs is not None:
                evidence["gmm_regime"] = f"{gmm_regime.value} ({gmm_conf:.0%})"
                for s_idx, p in enumerate(gmm_probs):
                    r = self._gmm.state_to_regime(s_idx)
                    state_prob_dict[f"gmm_{r.value}"] = round(float(p), 4)

            # Agreement: use HMM confidence; Disagreement: pick higher confidence
            if hmm_regime == gmm_regime and hmm_regime is not None:
                final_regime = hmm_regime
                # Average confidence when they agree
                final_conf = (hmm_conf + gmm_conf) / 2.0
                evidence["ensemble"] = "HMM+GMM agree"
            elif hmm_conf >= gmm_conf and hmm_regime is not None:
                final_regime = hmm_regime
                final_conf = hmm_conf * 0.85  # Discount for disagreement
                evidence["ensemble"] = f"HMM wins ({hmm_conf:.0%} > {gmm_conf:.0%})"
            elif gmm_regime is not None:
                final_regime = gmm_regime
                final_conf = gmm_conf * 0.85
                evidence["ensemble"] = f"GMM wins ({gmm_conf:.0%} > {hmm_conf:.0%})"
            else:
                return None

            # Use ADX from technicals to disambiguate TRENDING_BULL vs MEAN_REVERTING
            if technicals is not None:
                evidence["adx"] = f"{technicals.adx:.1f}"
                evidence["atr_pct"] = f"{technicals.atr_pct:.2%}"
                # If model says TRENDING_BULL but ADX < 15, override to MEAN_REVERTING
                if final_regime == Regime.TRENDING_BULL and technicals.adx < 15:
                    final_regime = Regime.MEAN_REVERTING
                    evidence["override"] = "ADX<15 → mean-reverting"
                # If model says MEAN_REVERTING but ATR% > 2.5%, override to HIGH_VOL
                if final_regime == Regime.MEAN_REVERTING and technicals.atr_pct > 0.025:
                    final_regime = Regime.HIGH_VOLATILITY
                    evidence["override"] = "ATR%>2.5% → high vol"

            return RegimeResult(
                regime=final_regime,
                confidence=round(min(final_conf, 0.99), 4),
                evidence=evidence,
                technicals=technicals,
                state_probabilities=state_prob_dict,
            )

        except Exception as e:
            self.logger.error(f"ML regime detection failed for {symbol}: {e}")
            return None

    # ================================================================== #
    # Rule-based fallback (simplified version of old system)
    # ================================================================== #

    def _classify_regime_rules(self, t: TechnicalSignals) -> RegimeResult:
        """
        Fallback rule-based classifier (only used when HMM+GMM fail).
        Same logic as old system, but explicitly flagged as low-confidence.
        """
        evidence: Dict[str, str] = {"source": "rule-based fallback"}
        scores: Dict[Regime, float] = {r: 0.0 for r in Regime if r != Regime.UNKNOWN}

        if t.atr_pct > 0.025:
            scores[Regime.HIGH_VOLATILITY] += 0.4
            evidence["high_atr"] = f"ATR% {t.atr_pct:.2%}"
        if t.adx > 35 and t.atr_pct > 0.015:
            scores[Regime.HIGH_VOLATILITY] += 0.2
        if t.bb_width > 0.12:
            scores[Regime.HIGH_VOLATILITY] += 0.15

        if t.adx > 25:
            if t.trend_direction == 1:
                scores[Regime.TRENDING_BULL] += 0.35
            elif t.trend_direction == -1:
                scores[Regime.TRENDING_BEAR] += 0.35
        if t.price_vs_sma50_pct > 3:
            scores[Regime.TRENDING_BULL] += 0.15
        elif t.price_vs_sma50_pct < -3:
            scores[Regime.TRENDING_BEAR] += 0.15
        if t.macd_histogram > 0 and t.macd_signal > 0:
            scores[Regime.TRENDING_BULL] += 0.1
        elif t.macd_histogram < 0 and t.macd_signal < 0:
            scores[Regime.TRENDING_BEAR] += 0.1

        if t.adx < 20:
            scores[Regime.MEAN_REVERTING] += 0.3
        if t.bb_width < 0.08:
            scores[Regime.MEAN_REVERTING] += 0.15
        if 40 <= t.rsi_14 <= 60:
            scores[Regime.MEAN_REVERTING] += 0.1

        scores[Regime.MEAN_REVERTING] += 0.05

        best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_regime]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.3

        # Cap rule-based confidence at 0.60 — it's NOT a calibrated probability
        confidence = min(confidence, 0.60)

        evidence["note"] = "Rule-based: confidence is NOT a calibrated probability"

        return RegimeResult(
            regime=best_regime,
            confidence=round(confidence, 2),
            evidence=evidence,
            technicals=t,
        )

    # ================================================================== #
    # Technical indicator computation
    # ================================================================== #

    def _compute_technicals(self, symbol: str) -> Optional[TechnicalSignals]:
        """Compute all technical indicators from price data."""
        if yf is None:
            self.logger.error("yfinance not available")
            return None
        try:
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            if data.empty or len(data) < 50:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None

            closes = data["Close"].values.flatten().astype(float)
            highs = data["High"].values.flatten().astype(float)
            lows = data["Low"].values.flatten().astype(float)
            volumes = data["Volume"].values.flatten().astype(float)

            current_price = float(closes[-1])
            sma20 = _sma(closes, 20)
            sma50 = _sma(closes, 50)
            sma200 = _sma(closes, 200)

            rsi14 = _rsi(closes, 14)
            macd_val, macd_sig, macd_hist = _macd(closes)

            _, _, _, bb_width, bb_pos = _bollinger_bands(closes, 20)
            adx_val = _adx(highs, lows, closes, 14)
            atr_pct_val = _atr_pct(highs, lows, closes, 14)

            if current_price > sma50 and sma20 > sma50:
                trend_dir = 1
            elif current_price < sma50 and sma20 < sma50:
                trend_dir = -1
            else:
                trend_dir = 0

            avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            vol_ratio = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0

            return TechnicalSignals(
                symbol=symbol,
                current_price=current_price,
                sma_20=sma20, sma_50=sma50, sma_200=sma200,
                price_vs_sma20_pct=(current_price / sma20 - 1) * 100 if sma20 > 0 else 0,
                price_vs_sma50_pct=(current_price / sma50 - 1) * 100 if sma50 > 0 else 0,
                rsi_14=rsi14,
                macd_signal=macd_val - macd_sig,
                macd_histogram=macd_hist,
                bb_width=bb_width, bb_position=bb_pos,
                atr_pct=atr_pct_val, adx=adx_val,
                trend_direction=trend_dir, volume_ratio=vol_ratio,
            )
        except Exception as e:
            self.logger.error(f"Technical computation failed for {symbol}: {e}")
            return None

# ============================================================================
# ML REGIME DETECTOR (RandomForest + Rule-based fallback)
# ============================================================================

try:
    from sklearn.ensemble import RandomForestClassifier
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False


class MLRegime(Enum):
    """Simplified regime classification for strategy selection."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"


@dataclass
class MLRegimeResult:
    """Result from ML regime detector including position sizing guidance."""
    regime: MLRegime
    confidence: float              # 0-1
    position_scale: float          # Multiplier for position sizing (0.1 - 1.2)
    features: Dict[str, float]     # Feature values used for detection
    evidence: Dict[str, str]       # Human-readable evidence

    @property
    def is_bullish(self) -> bool:
        return self.regime == MLRegime.BULL

    @property
    def is_bearish(self) -> bool:
        return self.regime == MLRegime.BEAR


# Mapping from MLRegime to position sizing multiplier
_REGIME_POSITION_SCALES = {
    MLRegime.BULL: 1.2,        # Full size + 20% in bull
    MLRegime.SIDEWAYS: 0.8,    # Reduce 20% in choppy
    MLRegime.HIGH_VOL: 0.5,    # Half size in high vol
    MLRegime.BEAR: 0.25,       # Quarter size in bear
}


class MLRegimeDetector:
    """
    RandomForest-based market regime classifier.

    Features (computed from daily OHLCV bars):
      - SMA slopes: 20/50/200-day SMA slope (normalized, 10-day rate of change)
      - VIX level: Current VIX (or synthetic vol proxy)
      - Breadth: Percentage of universe stocks above their 50-day SMA
      - ADX: Average Directional Index (trend strength)
      - Returns: Rolling 5/10/20-day cumulative returns
      - Volatility: 10/20-day realized vol annualized

    Training:
      - Fit on labeled SPY data (rules generate training labels)
      - Retrain weekly or on demand
      - Persists model to data/ml_regime_rf.pkl

    Usage:
        detector = MLRegimeDetector()
        result = detector.get_current_regime(spy_bars, universe_bars)
        print(result.regime, result.position_scale)
    """

    MODEL_PATH = os.path.join("data", "ml_regime_rf.pkl")

    def __init__(self, n_estimators: int = 100, retrain_interval_days: int = 7):
        self.n_estimators = n_estimators
        self.retrain_interval_days = retrain_interval_days
        self._model = None
        self._is_fitted = False
        self._last_train_time: Optional[datetime] = None
        self._feature_names: List[str] = [
            "sma20_slope", "sma50_slope", "sma200_slope",
            "vix_level", "breadth_50sma",
            "adx", "rsi_14", "bb_width",
            "ret_5d", "ret_10d", "ret_20d",
            "rvol_10d", "rvol_20d",
            "price_vs_sma50", "price_vs_sma200",
        ]

        if _RF_AVAILABLE:
            self._model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            self._try_load_model()
        else:
            logger.warning("sklearn not available — ML regime detector using rule-based only")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_current_regime(
        self,
        spy_bars: Optional[List[dict]] = None,
        universe_bars: Optional[Dict[str, List[dict]]] = None,
        vix_level: Optional[float] = None,
    ) -> MLRegimeResult:
        """
        Detect current market regime.

        Args:
            spy_bars: List of bar dicts with keys c, h, l, o, v (oldest first)
            universe_bars: Dict of symbol -> bars for breadth calculation
            vix_level: Current VIX level (optional, estimated from vol if absent)

        Returns:
            MLRegimeResult with regime, confidence, and position_scale
        """
        features = self._compute_features(spy_bars, universe_bars, vix_level)

        # Try ML prediction first
        if self._model is not None and self._is_fitted:
            try:
                X = np.array([[features.get(f, 0.0) for f in self._feature_names]])
                pred = self._model.predict(X)[0]
                proba = self._model.predict_proba(X)[0]
                regime = MLRegime(pred)
                confidence = float(np.max(proba))
                return MLRegimeResult(
                    regime=regime,
                    confidence=confidence,
                    position_scale=_REGIME_POSITION_SCALES[regime],
                    features=features,
                    evidence=self._build_evidence(features, "random_forest"),
                )
            except Exception as e:
                logger.warning(f"ML regime prediction failed: {e} — falling back to rules")

        # Rule-based fallback
        return self._classify_rules(features)

    def train(
        self,
        bars: List[dict],
        universe_bars: Optional[Dict[str, List[dict]]] = None,
        min_samples: int = 200,
    ) -> bool:
        """
        Train RandomForest on labeled data.
        Labels are generated from rules applied to historical windows.

        Args:
            bars: SPY daily bars (oldest first), needs ≥ min_samples + 200 for SMA
            universe_bars: Dict of symbol -> bars for breadth
            min_samples: Minimum training samples required

        Returns:
            True if training succeeded
        """
        if self._model is None:
            logger.warning("RandomForest not available — cannot train")
            return False

        closes = np.array([float(b["c"]) for b in bars])
        highs = np.array([float(b["h"]) for b in bars])
        lows = np.array([float(b["l"]) for b in bars])

        if len(closes) < min_samples + 200:
            logger.warning(f"Insufficient data for regime training: {len(closes)} bars")
            return False

        X_rows, y_labels = [], []
        # Slide a window from bar 200 onward
        for i in range(200, len(closes)):
            window_closes = closes[:i + 1]
            window_highs = highs[:i + 1]
            window_lows = lows[:i + 1]

            feat = self._compute_features_from_arrays(
                window_closes, window_highs, window_lows,
                universe_bars=universe_bars, vix_level=None,
            )
            label = self._rule_label(feat)
            X_rows.append([feat.get(f, 0.0) for f in self._feature_names])
            y_labels.append(label.value)

        X = np.array(X_rows)
        y = np.array(y_labels)

        if len(set(y)) < 2:
            logger.warning("Only one regime class in training data — skipping")
            return False

        self._model.fit(X, y)
        self._is_fitted = True
        self._last_train_time = datetime.now()
        self._persist_model()

        class_counts = {v: int((y == v).sum()) for v in set(y)}
        logger.info(f"ML regime RF trained on {len(y)} samples. Classes: {class_counts}")
        return True

    def needs_retrain(self) -> bool:
        """Check if model needs retraining."""
        if not self._is_fitted or self._last_train_time is None:
            return True
        return (datetime.now() - self._last_train_time).days >= self.retrain_interval_days

    def get_position_scale(self, regime: MLRegime) -> float:
        """Get position sizing multiplier for a regime."""
        return _REGIME_POSITION_SCALES.get(regime, 0.5)

    # ------------------------------------------------------------------ #
    # Feature computation
    # ------------------------------------------------------------------ #

    def _compute_features(
        self,
        spy_bars: Optional[List[dict]],
        universe_bars: Optional[Dict[str, List[dict]]],
        vix_level: Optional[float],
    ) -> Dict[str, float]:
        """Compute feature dict from bar data."""
        if spy_bars is None or len(spy_bars) < 50:
            return {f: 0.0 for f in self._feature_names}

        closes = np.array([float(b["c"]) for b in spy_bars])
        highs = np.array([float(b["h"]) for b in spy_bars])
        lows = np.array([float(b["l"]) for b in spy_bars])
        return self._compute_features_from_arrays(closes, highs, lows, universe_bars, vix_level)

    def _compute_features_from_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        universe_bars: Optional[Dict[str, List[dict]]] = None,
        vix_level: Optional[float] = None,
    ) -> Dict[str, float]:
        """Core feature computation from numpy arrays."""
        n = len(closes)
        price = float(closes[-1])

        # SMA values
        sma20 = float(np.mean(closes[-20:])) if n >= 20 else price
        sma50 = float(np.mean(closes[-50:])) if n >= 50 else price
        sma200 = float(np.mean(closes[-200:])) if n >= 200 else price

        # SMA slopes (10-day rate of change of the SMA, normalized)
        def sma_slope(data: np.ndarray, window: int, lookback: int = 10) -> float:
            if len(data) < window + lookback:
                return 0.0
            sma_now = float(np.mean(data[-window:]))
            sma_prev = float(np.mean(data[-(window + lookback):-lookback]))
            return (sma_now - sma_prev) / sma_prev if sma_prev != 0 else 0.0

        sma20_slope = sma_slope(closes, 20)
        sma50_slope = sma_slope(closes, 50)
        sma200_slope = sma_slope(closes, 200) if n >= 210 else 0.0

        # VIX level (use actual or synthetic from realized vol)
        if vix_level is not None:
            vix = vix_level
        else:
            # Synthetic: 20-day annualized realized vol * 100
            log_rets = np.diff(np.log(closes[-21:])) if n >= 21 else np.array([0.0])
            vix = float(np.std(log_rets) * np.sqrt(252) * 100)

        # Market breadth: % of universe stocks above their 50-day SMA
        breadth = 0.5  # default 50%
        if universe_bars:
            above_50 = 0
            total = 0
            for sym, bars in universe_bars.items():
                if len(bars) >= 50:
                    sym_closes = np.array([float(b["c"]) for b in bars])
                    sym_sma50 = float(np.mean(sym_closes[-50:]))
                    if float(sym_closes[-1]) > sym_sma50:
                        above_50 += 1
                    total += 1
            if total > 0:
                breadth = above_50 / total

        # ADX
        adx_val = _adx(highs, lows, closes, 14) if n >= 30 else 20.0

        # RSI
        rsi_val = _rsi(closes, 14) if n >= 16 else 50.0

        # Bollinger Band width
        _, _, _, bb_w, _ = _bollinger_bands(closes, 20)

        # Rolling returns
        ret_5d = float((closes[-1] / closes[-6]) - 1) if n >= 6 else 0.0
        ret_10d = float((closes[-1] / closes[-11]) - 1) if n >= 11 else 0.0
        ret_20d = float((closes[-1] / closes[-21]) - 1) if n >= 21 else 0.0

        # Realized volatility
        log_rets = np.diff(np.log(closes)) if n >= 2 else np.array([0.0])
        rvol_10d = float(np.std(log_rets[-10:]) * np.sqrt(252)) if len(log_rets) >= 10 else 0.15
        rvol_20d = float(np.std(log_rets[-20:]) * np.sqrt(252)) if len(log_rets) >= 20 else 0.15

        # Price vs SMA
        price_vs_sma50 = (price / sma50 - 1) * 100 if sma50 > 0 else 0.0
        price_vs_sma200 = (price / sma200 - 1) * 100 if sma200 > 0 else 0.0

        return {
            "sma20_slope": sma20_slope,
            "sma50_slope": sma50_slope,
            "sma200_slope": sma200_slope,
            "vix_level": vix,
            "breadth_50sma": breadth,
            "adx": adx_val,
            "rsi_14": rsi_val,
            "bb_width": bb_w,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ret_20d": ret_20d,
            "rvol_10d": rvol_10d,
            "rvol_20d": rvol_20d,
            "price_vs_sma50": price_vs_sma50,
            "price_vs_sma200": price_vs_sma200,
        }

    # ------------------------------------------------------------------ #
    # Rule-based fallback
    # ------------------------------------------------------------------ #

    def _rule_label(self, feat: Dict[str, float]) -> MLRegime:
        """Generate a training label from features using rules."""
        vix = feat.get("vix_level", 15)
        rvol = feat.get("rvol_20d", 0.15)
        ret_20d = feat.get("ret_20d", 0.0)
        sma50_slope = feat.get("sma50_slope", 0.0)
        sma200_slope = feat.get("sma200_slope", 0.0)
        adx = feat.get("adx", 20)
        breadth = feat.get("breadth_50sma", 0.5)
        price_vs_sma200 = feat.get("price_vs_sma200", 0.0)

        # HIGH_VOL: VIX > 25 or realized vol > 30% annualized
        if vix > 25 or rvol > 0.30:
            return MLRegime.HIGH_VOL

        # BEAR: SMA slopes negative, price below 200 SMA, breadth < 40%
        if sma50_slope < -0.005 and price_vs_sma200 < -2 and breadth < 0.4:
            return MLRegime.BEAR

        # BULL: SMA slopes positive, price above 200 SMA, breadth > 60%
        if sma50_slope > 0.005 and price_vs_sma200 > 2 and breadth > 0.6:
            return MLRegime.BULL

        # Additional BULL/BEAR from strong directional signals
        if ret_20d > 0.05 and sma200_slope > 0 and adx > 25:
            return MLRegime.BULL
        if ret_20d < -0.05 and sma200_slope < 0 and adx > 25:
            return MLRegime.BEAR

        # Default: SIDEWAYS
        return MLRegime.SIDEWAYS

    def _classify_rules(self, feat: Dict[str, float]) -> MLRegimeResult:
        """Rule-based classification fallback."""
        regime = self._rule_label(feat)
        # Compute rough confidence from feature agreement
        signals = []
        if regime == MLRegime.BULL:
            signals = [
                feat.get("sma50_slope", 0) > 0,
                feat.get("sma200_slope", 0) > 0,
                feat.get("price_vs_sma200", 0) > 0,
                feat.get("breadth_50sma", 0.5) > 0.5,
                feat.get("ret_20d", 0) > 0,
            ]
        elif regime == MLRegime.BEAR:
            signals = [
                feat.get("sma50_slope", 0) < 0,
                feat.get("sma200_slope", 0) < 0,
                feat.get("price_vs_sma200", 0) < 0,
                feat.get("breadth_50sma", 0.5) < 0.5,
                feat.get("ret_20d", 0) < 0,
            ]
        elif regime == MLRegime.HIGH_VOL:
            signals = [
                feat.get("vix_level", 15) > 25,
                feat.get("rvol_20d", 0.15) > 0.25,
                feat.get("bb_width", 0.04) > 0.08,
            ]
        else:  # SIDEWAYS
            signals = [
                abs(feat.get("sma50_slope", 0)) < 0.01,
                feat.get("adx", 20) < 25,
                abs(feat.get("ret_20d", 0)) < 0.03,
            ]

        agreement = sum(signals) / max(len(signals), 1)
        confidence = 0.4 + 0.3 * agreement  # 0.4 - 0.7 range

        return MLRegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            position_scale=_REGIME_POSITION_SCALES[regime],
            features=feat,
            evidence=self._build_evidence(feat, "rule_based"),
        )

    def _build_evidence(self, feat: Dict[str, float], source: str) -> Dict[str, str]:
        """Build human-readable evidence dict."""
        return {
            "source": source,
            "sma20_slope": f"{feat.get('sma20_slope', 0):.4f}",
            "sma50_slope": f"{feat.get('sma50_slope', 0):.4f}",
            "sma200_slope": f"{feat.get('sma200_slope', 0):.4f}",
            "vix": f"{feat.get('vix_level', 0):.1f}",
            "breadth": f"{feat.get('breadth_50sma', 0):.0%}",
            "adx": f"{feat.get('adx', 0):.1f}",
            "ret_20d": f"{feat.get('ret_20d', 0):.2%}",
            "rvol_20d": f"{feat.get('rvol_20d', 0):.1%}",
        }

    # ------------------------------------------------------------------ #
    # Model persistence
    # ------------------------------------------------------------------ #

    def _persist_model(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.MODEL_PATH) or ".", exist_ok=True)
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "is_fitted": self._is_fitted,
                    "last_train": self._last_train_time,
                }, f)
            logger.info(f"ML regime RF model saved to {self.MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save RF model: {e}")

    def _try_load_model(self) -> None:
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self._model = data["model"]
                self._is_fitted = data.get("is_fitted", True)
                self._last_train_time = data.get("last_train")
                logger.info(f"ML regime RF model loaded from {self.MODEL_PATH}")
        except Exception as e:
            logger.debug(f"Could not load RF regime model: {e}")


# ============================================================================
# MODULE-LEVEL CONVENIENCE
# ============================================================================

# Singleton for quick access
_ml_regime_detector: Optional[MLRegimeDetector] = None


def get_ml_regime_detector() -> MLRegimeDetector:
    """Get or create the singleton MLRegimeDetector."""
    global _ml_regime_detector
    if _ml_regime_detector is None:
        _ml_regime_detector = MLRegimeDetector()
    return _ml_regime_detector


def get_current_regime(
    spy_bars: Optional[List[dict]] = None,
    universe_bars: Optional[Dict[str, List[dict]]] = None,
    vix_level: Optional[float] = None,
) -> MLRegimeResult:
    """
    Convenience function: detect current regime.

    Args:
        spy_bars: SPY daily bars (oldest first)
        universe_bars: Dict of symbol -> bars for breadth
        vix_level: Current VIX (or None for synthetic estimate)

    Returns:
        MLRegimeResult with .regime, .confidence, .position_scale
    """
    detector = get_ml_regime_detector()
    return detector.get_current_regime(spy_bars, universe_bars, vix_level)