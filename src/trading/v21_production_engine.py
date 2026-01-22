"""
V2.1 Production Trading Engine
==============================

Production-ready trading engine combining proven V2.0 enhancements:
- ✅ Ensemble Regime Detection (+0.22 Sharpe contribution)
- ✅ Transformer Predictor (+0.09 Sharpe contribution)
- ✅ V1.3's proven TDA components
- ✅ Risk-adjusted position sizing
- ✅ Execution alpha optimization

Target: Sharpe > 1.55 (V1.3's 1.35 + proven enhancements)

Architecture:
    Data → TDA Features → [Ensemble Regime | Transformer] → Signals → Risk Sizing → Execution
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class V21Config:
    """Configuration for V2.1 Production Engine."""
    
    # Ensemble Regime Detection weights (must sum to 1.0)
    hmm_weight: float = 0.50
    gmm_weight: float = 0.30
    cluster_weight: float = 0.20
    
    # Transformer architecture
    transformer_d_model: int = 512
    transformer_n_heads: int = 8
    transformer_n_layers: int = 3
    transformer_dropout: float = 0.1
    transformer_sequence_length: int = 60
    
    # LSTM fallback (V1.3)
    lstm_units: int = 64
    lstm_sequence_length: int = 15
    
    # Position sizing
    max_position_pct: float = 0.03  # 3% max per position
    min_position_pct: float = 0.005  # 0.5% minimum position
    max_portfolio_heat: float = 0.20  # 20% max total risk
    risk_off_cash_pct: float = 0.50  # 50% cash in risk_off regime
    
    # Risk management
    stop_loss_sigma: float = 3.5
    max_drawdown_halt: float = 0.08
    circuit_breaker_dd: float = 0.05
    consecutive_loss_halt: int = 3
    
    # Feature toggles
    use_ensemble_regime: bool = True
    use_transformer: bool = True
    fallback_to_v13: bool = True
    
    # Universe
    universe_mode: str = "mega"  # "core", "expanded", "mega"
    max_positions: int = 50
    
    # Data
    data_lookback_days: int = 252 * 3  # 3 years
    tda_window: int = 20
    
    def validate(self) -> bool:
        """Validate configuration."""
        weights_sum = self.hmm_weight + self.gmm_weight + self.cluster_weight
        if abs(weights_sum - 1.0) > 0.01:
            logger.warning(f"Regime weights sum to {weights_sum}, normalizing...")
            self.hmm_weight /= weights_sum
            self.gmm_weight /= weights_sum
            self.cluster_weight /= weights_sum
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class V21ProductionEngine:
    """
    V2.1 Production Trading Engine
    
    Integrates proven V2.0 enhancements with V1.3's stable base.
    """
    
    def __init__(self, config: Optional[V21Config] = None):
        """Initialize V2.1 production engine."""
        self.config = config or V21Config()
        self.config.validate()
        
        # Component references
        self._ensemble_regime = None
        self._transformer = None
        self._lstm_predictor = None
        self._tda_generator = None
        
        # State tracking
        self._current_regime = "neutral"
        self._regime_confidence = 0.5
        self._consecutive_losses = 0
        self._is_halted = False
        self._halt_reason = ""
        
        # Data cache
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._tda_cache: Dict[str, np.ndarray] = {}
        self._last_cache_update: Optional[datetime] = None
        
        # Universe
        self._universe: List[str] = []
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all V2.1 components."""
        logger.info("Initializing V2.1 Production Engine...")
        
        # 1. Ensemble Regime Detection
        if self.config.use_ensemble_regime:
            try:
                from src.trading.regime_ensemble import EnsembleRegimeDetector
                self._ensemble_regime = EnsembleRegimeDetector(n_regimes=3)
                logger.info(f"✅ Ensemble Regime initialized (HMM:{self.config.hmm_weight:.0%}, "
                           f"GMM:{self.config.gmm_weight:.0%}, Cluster:{self.config.cluster_weight:.0%})")
            except Exception as e:
                logger.warning(f"⚠️ Ensemble Regime failed: {e}")
                self._ensemble_regime = None
                
        # 2. Transformer Predictor
        if self.config.use_transformer:
            try:
                from src.ml.transformer_predictor import TransformerPredictor
                self._transformer = TransformerPredictor(
                    d_model=self.config.transformer_d_model,
                    n_heads=self.config.transformer_n_heads,
                    n_layers=self.config.transformer_n_layers,
                    dropout=self.config.transformer_dropout,
                    sequence_length=self.config.transformer_sequence_length,
                )
                logger.info(f"✅ Transformer Predictor initialized "
                           f"(d_model={self.config.transformer_d_model})")
            except Exception as e:
                logger.warning(f"⚠️ Transformer failed: {e}")
                self._transformer = None
                
        # 3. LSTM Fallback (V1.3)
        if self.config.fallback_to_v13:
            try:
                from src.nn_predictor import NeuralNetPredictor
                self._lstm_predictor = NeuralNetPredictor(
                    sequence_length=self.config.lstm_sequence_length,
                    n_features=12,  # V1.2 TDA features
                    lstm_units=self.config.lstm_units,
                )
                logger.info("✅ LSTM Predictor (V1.3 fallback) initialized")
            except Exception as e:
                logger.warning(f"⚠️ LSTM Predictor failed: {e}")
                self._lstm_predictor = None
                
        # 4. TDA Feature Generator
        try:
            from src.tda_features import TDAFeatureGenerator
            self._tda_generator = TDAFeatureGenerator(
                window=self.config.tda_window,
                embedding_dim=3,
                feature_mode='v1.2',  # Use proven v1.2 features
            )
            logger.info("✅ TDA Feature Generator initialized")
        except Exception as e:
            logger.warning(f"⚠️ TDA Generator failed: {e}")
            self._tda_generator = None
            
        # 5. Load universe
        self._load_universe()
        
        logger.info("=" * 50)
        logger.info("V2.1 Engine initialization complete")
        logger.info(f"Components: Regime={self._ensemble_regime is not None}, "
                   f"Transformer={self._transformer is not None}, "
                   f"LSTM={self._lstm_predictor is not None}, "
                   f"TDA={self._tda_generator is not None}")
        logger.info(f"Universe: {len(self._universe)} symbols")
        logger.info("=" * 50)
        
    def _load_universe(self):
        """Load trading universe based on config."""
        if self.config.universe_mode == "mega":
            try:
                from src.trading.mega_universe import MEGA_UNIVERSE
                self._universe = MEGA_UNIVERSE[:700]  # Cap at 700
                logger.info(f"Loaded MEGA universe: {len(self._universe)} symbols")
            except ImportError:
                self._universe = self._get_fallback_universe()
        elif self.config.universe_mode == "expanded":
            try:
                from src.trading.full_universe import FULL_UNIVERSE
                self._universe = FULL_UNIVERSE
            except ImportError:
                self._universe = self._get_fallback_universe()
        else:  # core
            self._universe = ["SPY", "QQQ", "IWM", "XLF", "XLK"]
            
    def _get_fallback_universe(self) -> List[str]:
        """Fallback universe if imports fail."""
        return [
            "SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLE",
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "AVGO", "CRM",
            "JPM", "V", "MA", "BAC", "GS", "UNH", "JNJ", "LLY", "PFE", "ABBV",
        ]
        
    def get_universe(self) -> List[str]:
        """Return current trading universe."""
        return self._universe.copy()
        
    def get_component_status(self) -> Dict[str, Any]:
        """Return status of all components."""
        return {
            "ensemble_regime": self._ensemble_regime is not None,
            "transformer": self._transformer is not None,
            "lstm_predictor": self._lstm_predictor is not None,
            "tda_generator": self._tda_generator is not None,
            "current_regime": self._current_regime,
            "regime_confidence": self._regime_confidence,
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "universe_size": len(self._universe),
        }
        
    # =========================================================================
    # DATA MANAGEMENT
    # =========================================================================
    
    def fetch_data(self, tickers: List[str], 
                   days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for tickers.
        
        Args:
            tickers: List of ticker symbols
            days: Number of days to fetch (default: config.data_lookback_days)
            
        Returns:
            Dict mapping ticker -> DataFrame with OHLCV data
        """
        days = days or self.config.data_lookback_days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        result = {}
        
        # Try Polygon first
        try:
            from src.data.data_provider import get_ohlcv_data
            for ticker in tickers:
                try:
                    df = get_ohlcv_data(
                        ticker,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        timeframe="1d",
                    )
                    if df is not None and len(df) > 0:
                        result[ticker] = df
                        self._price_cache[ticker] = df
                except Exception as e:
                    logger.debug(f"Polygon failed for {ticker}: {e}")
        except ImportError:
            pass
            
        # Fallback to yfinance
        if len(result) < len(tickers):
            try:
                import yfinance as yf
                missing = [t for t in tickers if t not in result]
                for ticker in missing:
                    try:
                        df = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            progress=False,
                        )
                        if df is not None and len(df) > 0:
                            result[ticker] = df
                            self._price_cache[ticker] = df
                    except Exception as e:
                        logger.debug(f"yfinance failed for {ticker}: {e}")
            except ImportError:
                pass
                
        self._last_cache_update = datetime.now()
        return result
        
    def check_data_freshness(self, ticker: str, 
                             max_age_minutes: int = 60) -> bool:
        """Check if cached data is fresh enough."""
        if ticker not in self._price_cache:
            return False
            
        if self._last_cache_update is None:
            return False
            
        age = datetime.now() - self._last_cache_update
        return age.total_seconds() / 60 < max_age_minutes
        
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current/latest price for ticker."""
        if ticker in self._price_cache:
            df = self._price_cache[ticker]
            if len(df) > 0:
                return float(df['Close'].iloc[-1])
                
        # Try to fetch fresh
        data = self.fetch_data([ticker], days=5)
        if ticker in data and len(data[ticker]) > 0:
            return float(data[ticker]['Close'].iloc[-1])
            
        return None
        
    # =========================================================================
    # TDA FEATURES
    # =========================================================================
    
    def compute_tda_features(self, tickers: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute TDA features for tickers.
        
        Returns:
            Dict mapping ticker -> feature array
        """
        if self._tda_generator is None:
            logger.warning("TDA generator not available")
            return {}
            
        result = {}
        
        # Ensure we have data
        for ticker in tickers:
            if ticker not in self._price_cache:
                self.fetch_data([ticker])
                
        for ticker in tickers:
            if ticker not in self._price_cache:
                continue
                
            try:
                df = self._price_cache[ticker]
                if len(df) < self.config.tda_window * 2:
                    continue
                    
                # Extract close prices
                close = df['Close'].values
                
                # Compute TDA features
                features = self._tda_generator.generate_features(close)
                
                if features is not None and len(features) > 0:
                    result[ticker] = features
                    self._tda_cache[ticker] = features
                    
            except Exception as e:
                logger.debug(f"TDA computation failed for {ticker}: {e}")
                
        return result
        
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    def detect_regime(self, prices: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Detect current market regime using ensemble method.
        
        Returns:
            Tuple of (regime_name, confidence)
            Regimes: 'bull', 'bear', 'neutral', 'risk_off'
        """
        # Default to SPY for market regime
        if prices is None:
            if "SPY" not in self._price_cache:
                self.fetch_data(["SPY"])
            if "SPY" in self._price_cache:
                prices = self._price_cache["SPY"]['Close'].values
            else:
                return "neutral", 0.5
                
        if self._ensemble_regime is None:
            return self._simple_regime_detection(prices)
            
        try:
            # Compute features for regime detection
            returns = np.diff(np.log(prices))
            volatility = np.std(returns[-20:]) * np.sqrt(252)
            
            # Get TDA features if available
            tda_features = None
            if "SPY" in self._tda_cache:
                tda_features = self._tda_cache["SPY"][-1] if len(self._tda_cache["SPY"]) > 0 else None
                
            # Combine features
            feature_vec = np.array([
                returns[-1] if len(returns) > 0 else 0,  # Last return
                np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5-day momentum
                np.mean(returns[-20:]) if len(returns) >= 20 else 0,  # 20-day momentum
                volatility,
                np.percentile(returns[-60:], 5) if len(returns) >= 60 else -0.02,  # Tail risk
            ])
            
            if tda_features is not None and len(tda_features) >= 2:
                feature_vec = np.append(feature_vec, tda_features[:2])
                
            # Get ensemble prediction
            regime, confidence = self._ensemble_regime.predict_regime(
                feature_vec.reshape(1, -1),
                weights={
                    'hmm': self.config.hmm_weight,
                    'gmm': self.config.gmm_weight,
                    'cluster': self.config.cluster_weight,
                }
            )
            
            self._current_regime = regime
            self._regime_confidence = confidence
            
            return regime, confidence
            
        except Exception as e:
            logger.warning(f"Ensemble regime detection failed: {e}")
            return self._simple_regime_detection(prices)
            
    def _simple_regime_detection(self, prices: np.ndarray) -> Tuple[str, float]:
        """Simple fallback regime detection."""
        if len(prices) < 50:
            return "neutral", 0.5
            
        # Calculate indicators
        returns = np.diff(np.log(prices))
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        current = prices[-1]
        volatility = np.std(returns[-20:]) * np.sqrt(252)
        
        # Determine regime
        if current > sma_20 > sma_50 and volatility < 0.20:
            regime = "bull"
            confidence = 0.7
        elif current < sma_20 < sma_50 and volatility > 0.25:
            regime = "bear"
            confidence = 0.7
        elif volatility > 0.35:
            regime = "risk_off"
            confidence = 0.8
        else:
            regime = "neutral"
            confidence = 0.5
            
        self._current_regime = regime
        self._regime_confidence = confidence
        
        return regime, confidence
        
    # =========================================================================
    # PREDICTIONS
    # =========================================================================
    
    def generate_predictions(self, tickers: List[str],
                             tda_features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for tickers using V2.1 ensemble.
        
        Uses Transformer if available, falls back to LSTM.
        
        Returns:
            Dict mapping ticker -> prediction info
        """
        predictions = {}
        
        for ticker in tickers:
            if ticker not in self._price_cache:
                continue
                
            try:
                pred = self._predict_single(ticker, tda_features.get(ticker))
                if pred is not None:
                    predictions[ticker] = pred
            except Exception as e:
                logger.debug(f"Prediction failed for {ticker}: {e}")
                
        return predictions
        
    def _predict_single(self, ticker: str, 
                        tda_features: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Generate prediction for single ticker."""
        df = self._price_cache.get(ticker)
        if df is None or len(df) < 60:
            return None
            
        # Prepare features
        close = df['Close'].values
        returns = np.diff(np.log(close))
        
        # Try Transformer first
        if self._transformer is not None:
            try:
                seq_len = self.config.transformer_sequence_length
                if len(returns) >= seq_len:
                    # Build feature sequence
                    features = self._build_feature_sequence(df, tda_features, seq_len)
                    
                    # Get transformer prediction
                    prob = self._transformer.predict(features)
                    
                    return {
                        "ticker": ticker,
                        "direction_prob": float(prob),
                        "direction": "long" if prob > 0.55 else ("short" if prob < 0.45 else "flat"),
                        "confidence": float(abs(prob - 0.5) * 2),
                        "predictor": "transformer",
                    }
            except Exception as e:
                logger.debug(f"Transformer prediction failed for {ticker}: {e}")
                
        # Fallback to LSTM
        if self._lstm_predictor is not None and self.config.fallback_to_v13:
            try:
                seq_len = self.config.lstm_sequence_length
                if len(returns) >= seq_len:
                    features = self._build_feature_sequence(df, tda_features, seq_len)
                    prob = self._lstm_predictor.predict(features)
                    
                    return {
                        "ticker": ticker,
                        "direction_prob": float(prob),
                        "direction": "long" if prob > 0.52 else ("short" if prob < 0.48 else "flat"),
                        "confidence": float(abs(prob - 0.5) * 2),
                        "predictor": "lstm",
                    }
            except Exception as e:
                logger.debug(f"LSTM prediction failed for {ticker}: {e}")
                
        # Simple momentum fallback
        mom_20 = np.mean(returns[-20:]) if len(returns) >= 20 else 0
        mom_5 = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        
        prob = 0.5 + (mom_20 * 50) + (mom_5 * 25)  # Simple momentum signal
        prob = np.clip(prob, 0, 1)
        
        return {
            "ticker": ticker,
            "direction_prob": float(prob),
            "direction": "long" if prob > 0.55 else ("short" if prob < 0.45 else "flat"),
            "confidence": float(abs(prob - 0.5) * 2),
            "predictor": "momentum",
        }
        
    def _build_feature_sequence(self, df: pd.DataFrame,
                                 tda_features: Optional[np.ndarray],
                                 seq_len: int) -> np.ndarray:
        """Build feature sequence for model input."""
        close = df['Close'].values[-seq_len-1:]
        high = df['High'].values[-seq_len-1:] if 'High' in df else close
        low = df['Low'].values[-seq_len-1:] if 'Low' in df else close
        volume = df['Volume'].values[-seq_len:] if 'Volume' in df else np.ones(seq_len)
        
        returns = np.diff(np.log(close))
        hl_range = (high[1:] - low[1:]) / close[1:]
        vol_change = np.diff(np.log(volume + 1))
        
        # Base features
        features = np.column_stack([
            returns,
            hl_range,
        ])
        
        # Add TDA features if available
        if tda_features is not None:
            # Broadcast TDA features across sequence
            if len(tda_features.shape) == 1:
                tda_broadcast = np.tile(tda_features, (seq_len, 1))
            else:
                tda_broadcast = tda_features[-seq_len:]
            features = np.column_stack([features, tda_broadcast[:len(features)]])
            
        return features.reshape(1, seq_len, -1)
        
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def compute_position_sizes(self, predictions: Dict[str, Dict[str, Any]],
                                regime: str,
                                regime_confidence: float,
                                max_position_pct: float = 0.03,
                                max_heat: float = 0.20) -> Dict[str, Dict[str, Any]]:
        """
        Compute risk-adjusted position sizes.
        
        Args:
            predictions: Prediction dict from generate_predictions
            regime: Current market regime
            regime_confidence: Regime detection confidence
            max_position_pct: Maximum position size (fraction of portfolio)
            max_heat: Maximum total portfolio risk
            
        Returns:
            Dict mapping ticker -> signal with position weight
        """
        signals = {}
        
        # Regime-based scaling
        if regime == "risk_off":
            regime_scale = 0.25
        elif regime == "bear":
            regime_scale = 0.5
        elif regime == "neutral":
            regime_scale = 0.75
        else:  # bull
            regime_scale = 1.0
            
        # Confidence scaling
        confidence_scale = 0.5 + (regime_confidence * 0.5)
        
        # Sort by confidence for allocation
        sorted_preds = sorted(
            predictions.items(),
            key=lambda x: x[1].get("confidence", 0),
            reverse=True
        )
        
        total_heat = 0.0
        
        for ticker, pred in sorted_preds:
            if total_heat >= max_heat:
                break
                
            direction = pred.get("direction", "flat")
            if direction == "flat":
                continue
                
            # Base weight from confidence
            confidence = pred.get("confidence", 0.5)
            base_weight = confidence * max_position_pct
            
            # Apply regime and confidence scaling
            weight = base_weight * regime_scale * confidence_scale
            
            # Ensure minimum position
            if weight < self.config.min_position_pct:
                continue
                
            # Cap at max position
            weight = min(weight, max_position_pct)
            
            # Check heat budget
            if total_heat + weight > max_heat:
                weight = max_heat - total_heat
                
            signals[ticker] = {
                **pred,
                "weight": weight,
                "regime": regime,
                "regime_confidence": regime_confidence,
            }
            
            total_heat += weight
            
            # Stop after max positions
            if len(signals) >= self.config.max_positions:
                break
                
        logger.info(f"Position sizing: {len(signals)} positions, "
                   f"total heat: {total_heat:.1%}, "
                   f"regime: {regime} ({regime_confidence:.1%})")
                   
        return signals
        
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    
    def check_risk_limits(self, current_dd: float, 
                          daily_pnl: float) -> Tuple[bool, str]:
        """
        Check if risk limits are breached.
        
        Returns:
            Tuple of (should_halt, reason)
        """
        if self._is_halted:
            return True, self._halt_reason
            
        # Emergency halt
        if current_dd >= self.config.max_drawdown_halt:
            self._is_halted = True
            self._halt_reason = f"Emergency halt: {current_dd:.1%} drawdown"
            return True, self._halt_reason
            
        # Circuit breaker warning
        if current_dd >= self.config.circuit_breaker_dd:
            logger.warning(f"Circuit breaker warning: {current_dd:.1%} drawdown")
            
        # Consecutive losses
        if daily_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
            
        if self._consecutive_losses >= self.config.consecutive_loss_halt:
            self._is_halted = True
            self._halt_reason = f"Consecutive loss halt: {self._consecutive_losses} days"
            return True, self._halt_reason
            
        return False, ""
        
    def reset_halt(self):
        """Reset halt state (manual override)."""
        self._is_halted = False
        self._halt_reason = ""
        self._consecutive_losses = 0
        logger.info("Trading halt reset manually")
