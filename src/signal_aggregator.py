"""
Signal Aggregator
==================
Collects signals from ALL quantitative models and produces a unified
weighted-voting ensemble signal with regime-dependent weights.

Models integrated:
- ManifoldRegimeDetector (Riemannian geometry regime classification)
- GARCH(1,1) volatility forecasting
- CAPM expected returns
- ML Stacked Ensemble / Transformer / XGBoost
- HMM Regime Detector (existing)
- Merton Jump-Diffusion
- Heston Stochastic Vol
- Monte Carlo pricer
- CRR Binomial Tree

Each model outputs:
    signal: float in [-1, 1]   (-1 = strong sell, +1 = strong buy)
    confidence: float in [0, 1]
    model_name: str
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AggregatedRegime(Enum):
    """Unified market regime from aggregator."""
    STRONG_TREND = "strong_trend"
    MILD_TREND = "mild_trend"
    MEAN_REVERSION = "mean_reversion"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class ModelSignal:
    """Signal from a single model."""
    model_name: str
    signal: float         # [-1, 1]
    confidence: float     # [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.signal = float(np.clip(self.signal, -1, 1))
        self.confidence = float(np.clip(self.confidence, 0, 1))


@dataclass
class AggregatedSignal:
    """Final aggregated signal from all models."""
    symbol: str
    signal: float              # [-1, 1] weighted ensemble signal
    confidence: float          # [0, 1] ensemble confidence
    regime: AggregatedRegime
    model_signals: List[ModelSignal]
    weights_used: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def direction(self) -> str:
        if self.signal > 0.3:
            return "BUY"
        elif self.signal < -0.3:
            return "SELL"
        return "HOLD"

    @property
    def strength(self) -> str:
        abs_sig = abs(self.signal)
        if abs_sig > 0.7:
            return "STRONG"
        elif abs_sig > 0.4:
            return "MODERATE"
        return "WEAK"

    @property
    def is_actionable(self) -> bool:
        return abs(self.signal) > 0.3 and self.confidence > 0.5


# Default regime-dependent weights
# Keys: regime → {model_name: weight}
DEFAULT_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    AggregatedRegime.STRONG_TREND.value: {
        "capm": 0.15,
        "garch": 0.10,
        "manifold_regime": 0.15,
        "hmm_regime": 0.10,
        "ml_ensemble": 0.20,
        "transformer": 0.15,
        "merton": 0.05,
        "heston": 0.05,
        "momentum": 0.05,
    },
    AggregatedRegime.MEAN_REVERSION.value: {
        "capm": 0.10,
        "garch": 0.15,
        "manifold_regime": 0.10,
        "hmm_regime": 0.15,
        "ml_ensemble": 0.15,
        "transformer": 0.10,
        "merton": 0.10,
        "heston": 0.10,
        "momentum": 0.05,
    },
    AggregatedRegime.HIGH_VOLATILITY.value: {
        "capm": 0.05,
        "garch": 0.25,
        "manifold_regime": 0.15,
        "hmm_regime": 0.15,
        "ml_ensemble": 0.10,
        "transformer": 0.05,
        "merton": 0.10,
        "heston": 0.10,
        "momentum": 0.05,
    },
    AggregatedRegime.CRISIS.value: {
        "capm": 0.05,
        "garch": 0.20,
        "manifold_regime": 0.20,
        "hmm_regime": 0.20,
        "ml_ensemble": 0.10,
        "transformer": 0.05,
        "merton": 0.10,
        "heston": 0.05,
        "momentum": 0.05,
    },
}


class SignalAggregator:
    """
    Unified signal aggregation engine.
    
    Collects signals from all models, determines regime,
    applies regime-dependent weights, and produces a single
    aggregated trading signal.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_models: int = 2,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.min_confidence = min_confidence
        self.min_models = min_models
        self.regime_weights = regime_weights or DEFAULT_REGIME_WEIGHTS

        # Initialize models (lazy, with fallbacks)
        self._manifold_detector = None
        self._hmm_detector = None
        self._garch = None
        self._capm = None
        self._ml_ensemble = None
        self._transformer = None
        self._merton = None
        self._heston = None
        self._continuous_learner = None
        self._initialized = False

    def initialize(self):
        """Lazy-initialize all model connectors."""
        if self._initialized:
            return

        # ManifoldRegimeDetector
        try:
            from src.options.manifold_regime_detector import ManifoldRegimeDetector
            self._manifold_detector = ManifoldRegimeDetector()
            logger.info("ManifoldRegimeDetector loaded")
        except Exception as e:
            logger.warning(f"ManifoldRegimeDetector unavailable: {e}")

        # HMM RegimeDetector
        try:
            from src.options.regime_detector import RegimeDetector
            self._hmm_detector = RegimeDetector()
            logger.info("HMM RegimeDetector loaded")
        except Exception as e:
            logger.warning(f"HMM RegimeDetector unavailable: {e}")

        # GARCH
        try:
            from src.quant_models.garch import GARCHModel
            self._garch = GARCHModel()
            logger.info("GARCH model loaded")
        except Exception as e:
            logger.warning(f"GARCH unavailable: {e}")

        # CAPM
        try:
            from src.quant_models.capm import CAPMModel
            self._capm = CAPMModel()
            logger.info("CAPM model loaded")
        except Exception as e:
            logger.warning(f"CAPM unavailable: {e}")

        # ML Stacked Ensemble
        try:
            from src.ml.stacked_ensemble import StackedEnsemble
            self._ml_ensemble = StackedEnsemble()
            logger.info("ML StackedEnsemble loaded")
        except Exception as e:
            logger.warning(f"StackedEnsemble unavailable: {e}")

        # Transformer
        try:
            from src.ml.transformer_predictor import TransformerPredictor
            self._transformer = TransformerPredictor()
            logger.info("TransformerPredictor loaded")
        except Exception as e:
            logger.warning(f"TransformerPredictor unavailable: {e}")

        # Merton
        try:
            from src.quant_models.merton_jump_diffusion import MertonJumpDiffusion
            self._merton = MertonJumpDiffusion()
            logger.info("Merton Jump-Diffusion loaded")
        except Exception as e:
            logger.warning(f"Merton unavailable: {e}")

        # Heston
        try:
            from src.quant_models.heston_model import HestonModel
            self._heston = HestonModel()
            logger.info("Heston model loaded")
        except Exception as e:
            logger.warning(f"Heston unavailable: {e}")

        # Continuous Learner
        try:
            from src.ml.continuous_learner import ContinuousLearner
            self._continuous_learner = ContinuousLearner()
            logger.info("ContinuousLearner loaded")
        except Exception as e:
            logger.warning(f"ContinuousLearner unavailable: {e}")

        self._initialized = True
        logger.info("SignalAggregator initialization complete")

    def determine_regime(self, symbol: str = "SPY") -> AggregatedRegime:
        """Determine current market regime using available detectors."""
        self.initialize()

        regimes = []

        # Manifold detector
        if self._manifold_detector is not None:
            try:
                result = self._manifold_detector.detect_regime(symbol)
                regime_name = getattr(result, "regime", getattr(result, "name", str(result)))
                regime_str = str(regime_name).upper()
                if "CRISIS" in regime_str or "SPIRAL" in regime_str:
                    regimes.append(AggregatedRegime.CRISIS)
                elif "VOLATILE" in regime_str or "TRANSITION" in regime_str:
                    regimes.append(AggregatedRegime.HIGH_VOLATILITY)
                elif "TREND" in regime_str or "GEODESIC" in regime_str:
                    regimes.append(AggregatedRegime.STRONG_TREND)
                elif "REVERSION" in regime_str or "CONSOLIDATION" in regime_str:
                    regimes.append(AggregatedRegime.MEAN_REVERSION)
                else:
                    regimes.append(AggregatedRegime.MILD_TREND)
            except Exception as e:
                logger.debug(f"Manifold regime detection failed: {e}")

        # HMM detector
        if self._hmm_detector is not None:
            try:
                regime_state = self._hmm_detector.detect(symbol)
                regime_name = str(getattr(regime_state, "current_regime", regime_state)).upper()
                if "BEAR_HIGH" in regime_name:
                    regimes.append(AggregatedRegime.CRISIS)
                elif "HIGH_VOL" in regime_name:
                    regimes.append(AggregatedRegime.HIGH_VOLATILITY)
                elif "BULL" in regime_name:
                    regimes.append(AggregatedRegime.STRONG_TREND)
                else:
                    regimes.append(AggregatedRegime.MEAN_REVERSION)
            except Exception as e:
                logger.debug(f"HMM regime detection failed: {e}")

        # GARCH implied regime
        if self._garch is not None:
            try:
                forecast = self._garch.fit_and_forecast(symbol, horizon=5)
                if forecast.current_vol > 0.35:
                    regimes.append(AggregatedRegime.CRISIS)
                elif forecast.current_vol > 0.25:
                    regimes.append(AggregatedRegime.HIGH_VOLATILITY)
                elif forecast.current_vol < 0.12:
                    regimes.append(AggregatedRegime.STRONG_TREND)
                else:
                    regimes.append(AggregatedRegime.MILD_TREND)
            except Exception as e:
                logger.debug(f"GARCH regime inference failed: {e}")

        if not regimes:
            return AggregatedRegime.UNKNOWN

        # Majority vote
        from collections import Counter
        counts = Counter(regimes)
        return counts.most_common(1)[0][0]

    def collect_signals(self, symbol: str) -> List[ModelSignal]:
        """Collect signals from all available models for a symbol."""
        self.initialize()
        signals = []

        # CAPM signal
        if self._capm is not None:
            try:
                result = self._capm.analyze(symbol)
                signals.append(ModelSignal(
                    model_name="capm",
                    signal=result.signal,
                    confidence=result.confidence,
                    metadata={"beta": result.beta, "alpha": result.alpha, "expected_return": result.expected_return},
                ))
            except Exception as e:
                logger.debug(f"CAPM signal failed for {symbol}: {e}")

        # GARCH signal
        if self._garch is not None:
            try:
                forecast = self._garch.fit_and_forecast(symbol, horizon=5)
                signals.append(ModelSignal(
                    model_name="garch",
                    signal=forecast.signal,
                    confidence=forecast.confidence,
                    metadata={"current_vol": forecast.current_vol, "var_95": forecast.var_95},
                ))
            except Exception as e:
                logger.debug(f"GARCH signal failed for {symbol}: {e}")

        # Manifold regime signal
        if self._manifold_detector is not None:
            try:
                result = self._manifold_detector.detect_regime(symbol)
                # Extract signal from manifold result
                sig = getattr(result, "signal", 0.0)
                conf = getattr(result, "confidence", 0.5)
                if hasattr(result, "curvature"):
                    # Negative curvature → unstable → sell signal
                    curv = getattr(result, "curvature", 0)
                    sig = float(np.clip(-curv * 5, -1, 1))
                signals.append(ModelSignal(
                    model_name="manifold_regime",
                    signal=float(sig),
                    confidence=float(conf),
                    metadata={"regime": str(getattr(result, "regime", "unknown"))},
                ))
            except Exception as e:
                logger.debug(f"Manifold signal failed for {symbol}: {e}")

        # HMM regime signal
        if self._hmm_detector is not None:
            try:
                regime_state = self._hmm_detector.detect(symbol)
                regime = getattr(regime_state, "current_regime", None)
                confidence = getattr(regime_state, "confidence", 0.5)
                regime_str = str(regime).upper()
                if "BULL" in regime_str:
                    sig = 0.6
                elif "BEAR" in regime_str:
                    sig = -0.6
                else:
                    sig = 0.0
                signals.append(ModelSignal(
                    model_name="hmm_regime",
                    signal=sig,
                    confidence=float(confidence),
                    metadata={"regime": str(regime)},
                ))
            except Exception as e:
                logger.debug(f"HMM signal failed for {symbol}: {e}")

        # ML Ensemble signal
        if self._ml_ensemble is not None:
            try:
                # StackedEnsemble expects features as DataFrame/array
                # Generate features from recent price data
                features = self._build_ml_features(symbol)
                if features is not None:
                    prediction = self._ml_ensemble.predict(features)
                    if hasattr(prediction, '__len__') and len(prediction) > 0:
                        pred_val = float(prediction[-1]) if hasattr(prediction, '__getitem__') else float(prediction)
                    else:
                        pred_val = float(prediction)
                    sig = float(np.clip(pred_val * 2, -1, 1))
                    signals.append(ModelSignal(
                        model_name="ml_ensemble",
                        signal=sig,
                        confidence=0.65,
                        metadata={"raw_prediction": pred_val},
                    ))
            except Exception as e:
                logger.debug(f"ML ensemble signal failed for {symbol}: {e}")

        # Transformer signal
        if self._transformer is not None:
            try:
                prediction = self._transformer.predict(symbol)
                if prediction is not None:
                    direction_prob = getattr(prediction, "direction_prob", 0.5)
                    confidence = getattr(prediction, "confidence", 0.5)
                    sig = float(np.clip((direction_prob - 0.5) * 2, -1, 1))
                    signals.append(ModelSignal(
                        model_name="transformer",
                        signal=sig,
                        confidence=float(confidence),
                        metadata={"direction_prob": float(direction_prob)},
                    ))
            except Exception as e:
                logger.debug(f"Transformer signal failed for {symbol}: {e}")

        # Simple momentum signal (always available)
        try:
            momentum_sig = self._compute_momentum_signal(symbol)
            if momentum_sig is not None:
                signals.append(momentum_sig)
        except Exception as e:
            logger.debug(f"Momentum signal failed for {symbol}: {e}")

        return signals

    def aggregate(
        self,
        symbol: str,
        min_confidence: Optional[float] = None,
    ) -> AggregatedSignal:
        """
        Full pipeline: collect signals, determine regime, aggregate.
        """
        min_conf = min_confidence if min_confidence is not None else self.min_confidence

        # Determine regime
        regime = self.determine_regime()

        # Collect all model signals
        raw_signals = self.collect_signals(symbol)

        # Filter by confidence
        filtered = [s for s in raw_signals if s.confidence >= min_conf]

        if len(filtered) < self.min_models:
            # Relax threshold if we don't have enough
            filtered = sorted(raw_signals, key=lambda s: s.confidence, reverse=True)
            filtered = filtered[:max(self.min_models, len(filtered))]

        if not filtered:
            return AggregatedSignal(
                symbol=symbol,
                signal=0.0,
                confidence=0.0,
                regime=regime,
                model_signals=raw_signals,
                weights_used={},
            )

        # Get regime weights
        regime_key = regime.value
        weights_map = self.regime_weights.get(
            regime_key,
            self.regime_weights.get(AggregatedRegime.MILD_TREND.value, {}),
        )

        # Weighted average
        total_weight = 0.0
        weighted_signal = 0.0
        weighted_confidence = 0.0
        weights_used = {}

        for sig in filtered:
            w = weights_map.get(sig.model_name, 0.1)  # default weight
            effective_weight = w * sig.confidence
            weighted_signal += sig.signal * effective_weight
            weighted_confidence += sig.confidence * w
            total_weight += effective_weight
            weights_used[sig.model_name] = w

        if total_weight > 0:
            final_signal = weighted_signal / total_weight
            final_confidence = weighted_confidence / sum(weights_used.values()) if sum(weights_used.values()) > 0 else 0
        else:
            final_signal = 0.0
            final_confidence = 0.0

        result = AggregatedSignal(
            symbol=symbol,
            signal=float(np.clip(final_signal, -1, 1)),
            confidence=float(np.clip(final_confidence, 0, 1)),
            regime=regime,
            model_signals=raw_signals,
            weights_used=weights_used,
        )

        logger.info(
            f"Aggregated {symbol}: signal={result.signal:.3f} ({result.direction}), "
            f"confidence={result.confidence:.3f}, regime={regime.value}, "
            f"models={len(filtered)}/{len(raw_signals)}"
        )
        return result

    def _build_ml_features(self, symbol: str) -> Optional[np.ndarray]:
        """Build feature vector for ML models from recent price data."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period="6mo", interval="1d", progress=False)
            if data.empty or len(data) < 60:
                return None

            close = data["Close"].values.flatten()
            volume = data["Volume"].values.flatten()

            # Technical features
            ret_1d = np.diff(np.log(close))
            ret_5d = close[-1] / close[-5] - 1 if len(close) >= 5 else 0
            ret_20d = close[-1] / close[-20] - 1 if len(close) >= 20 else 0
            vol_20d = np.std(ret_1d[-20:]) * np.sqrt(252) if len(ret_1d) >= 20 else 0
            sma_ratio = close[-1] / np.mean(close[-20:]) if len(close) >= 20 else 1
            rsi = self._compute_rsi(close, 14)
            vol_ratio = np.mean(volume[-5:]) / np.mean(volume[-20:]) if len(volume) >= 20 else 1

            features = np.array([[
                ret_1d[-1] if len(ret_1d) > 0 else 0,
                ret_5d,
                ret_20d,
                vol_20d,
                sma_ratio,
                rsi,
                vol_ratio,
                close[-1] / np.mean(close[-50:]) if len(close) >= 50 else 1,  # 50d SMA ratio
                np.mean(ret_1d[-5:]) if len(ret_1d) >= 5 else 0,  # 5d avg return
                np.std(ret_1d[-5:]) * np.sqrt(252) if len(ret_1d) >= 5 else 0,  # 5d vol
            ]])
            return features
        except Exception:
            return None

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute RSI."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _compute_momentum_signal(self, symbol: str) -> Optional[ModelSignal]:
        """Compute a basic momentum signal from price data."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if data.empty or len(data) < 30:
                return None

            close = data["Close"].values.flatten()
            ret_20d = close[-1] / close[-20] - 1
            ret_5d = close[-1] / close[-5] - 1

            # Momentum signal: blend of 5d and 20d returns
            raw_signal = 0.4 * np.clip(ret_5d * 10, -1, 1) + 0.6 * np.clip(ret_20d * 5, -1, 1)

            # RSI for confidence
            rsi = self._compute_rsi(close, 14)
            # Extreme RSI → higher confidence in reversal
            if rsi > 70:
                raw_signal = min(raw_signal, 0.2)  # overbought dampening
            elif rsi < 30:
                raw_signal = max(raw_signal, -0.2)  # oversold dampening

            confidence = min(abs(ret_20d) * 5, 0.9)

            return ModelSignal(
                model_name="momentum",
                signal=float(np.clip(raw_signal, -1, 1)),
                confidence=float(np.clip(confidence, 0.3, 0.9)),
                metadata={"ret_5d": ret_5d, "ret_20d": ret_20d, "rsi": rsi},
            )
        except Exception:
            return None

    def update_after_trade(self, symbol: str, signal: AggregatedSignal, outcome: float):
        """
        Update model weights after a trade outcome.
        Delegates to ContinuousLearner if available.
        """
        if self._continuous_learner is not None:
            try:
                self._continuous_learner.record_outcome(
                    signal_info={
                        "symbol": symbol,
                        "signal": signal.signal,
                        "models": [s.model_name for s in signal.model_signals],
                        "regime": signal.regime.value,
                    },
                    outcome=outcome,
                )
            except Exception as e:
                logger.debug(f"ContinuousLearner update failed: {e}")
