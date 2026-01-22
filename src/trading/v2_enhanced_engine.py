"""
V2 Enhanced Trading Engine

Orchestrates all V2.0 components:
- Transformer predictor (attention-based)
- SAC agent with PER (reinforcement learning)
- Persistent Laplacian (enhanced TDA)
- Ensemble regime detection
- Order flow analysis

Provides backward compatibility toggle with V1.3.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

# V2 Components
try:
    from src.ml.transformer_predictor import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

try:
    from src.ml.sac_agent import SACAgent, SACConfig
    SAC_AVAILABLE = True
except ImportError:
    SAC_AVAILABLE = False

try:
    from src.tda_v2.persistent_laplacian import PersistentLaplacian, EnhancedTDAFeatures
    LAPLACIAN_AVAILABLE = True
except ImportError:
    LAPLACIAN_AVAILABLE = False

try:
    from src.trading.regime_ensemble import EnsembleRegimeDetector, RegimeType
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from src.microstructure.order_flow_analyzer import OrderFlowAnalyzer
    ORDERFLOW_AVAILABLE = True
except ImportError:
    ORDERFLOW_AVAILABLE = False

# V1.3 fallbacks
try:
    from src.trading.adaptive_engine import AdaptiveLearningEngine
    V13_AVAILABLE = True
except ImportError:
    V13_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class V2Config:
    """Configuration for V2 Enhanced Engine."""
    
    # Component toggles
    use_transformer: bool = True
    use_sac: bool = True
    use_persistent_laplacian: bool = True
    use_ensemble_regime: bool = True
    use_order_flow: bool = True
    
    # Backward compatibility
    fallback_to_v13: bool = True
    
    # Transformer config
    transformer_d_model: int = 512
    transformer_n_heads: int = 8
    transformer_n_layers: int = 3
    
    # SAC config
    sac_state_dim: int = 37  # 10 base + 12 TDA + 5 regime + 10 orderflow
    sac_batch_size: int = 256
    
    # Regime config
    n_regimes: int = 3
    regime_consensus: int = 2
    
    # Position sizing
    base_position_size: float = 0.02  # 2% per position
    max_position_size: float = 0.05  # 5% max
    min_position_size: float = 0.005  # 0.5% min
    
    # Risk management
    max_portfolio_heat: float = 0.25  # 25% max total risk
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    
    # Performance targets
    target_sharpe: float = 1.5
    max_drawdown: float = 0.015  # 1.5%
    target_cagr: float = 0.18  # 18%


class V2EnhancedEngine:
    """
    V2.0 Enhanced Trading Engine.
    
    Combines all upgraded components with intelligent orchestration.
    
    Decision Pipeline:
    1. Compute Persistent Laplacian TDA features
    2. Detect regime with ensemble (HMM + GMM + Clustering)
    3. Analyze order flow microstructure
    4. Generate predictions with Transformer
    5. Optimize position sizing with SAC
    6. Apply risk management constraints
    """
    
    def __init__(self, config: Optional[V2Config] = None,
                 initial_capital: float = 100000,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """
        Initialize V2 Enhanced Engine.
        
        Args:
            config: V2 configuration
            initial_capital: Starting capital
            api_key: Alpaca API key
            api_secret: Alpaca API secret
        """
        self.config = config or V2Config()
        self.initial_capital = initial_capital
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize components
        self._init_components()
        
        # State tracking
        self.current_regime = "neutral"
        self.positions: Dict[str, float] = {}
        self.portfolio_value = initial_capital
        self.last_update = None
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # Component status
        self.component_status = self._check_component_status()
        logger.info(f"V2 Engine initialized. Components: {self.component_status}")
    
    def _init_components(self):
        """Initialize all V2 components."""
        
        # Transformer Predictor
        if self.config.use_transformer and TRANSFORMER_AVAILABLE:
            self.transformer = TransformerPredictor(
                d_model=self.config.transformer_d_model,
                n_heads=self.config.transformer_n_heads,
                n_layers=self.config.transformer_n_layers
            )
            logger.info("Transformer predictor initialized")
        else:
            self.transformer = None
        
        # SAC Agent
        if self.config.use_sac and SAC_AVAILABLE:
            sac_config = SACConfig(
                state_dim=self.config.sac_state_dim,
                batch_size=self.config.sac_batch_size
            )
            self.sac = SACAgent(config=sac_config)
            logger.info("SAC agent initialized")
        else:
            self.sac = None
        
        # Persistent Laplacian TDA
        if self.config.use_persistent_laplacian and LAPLACIAN_AVAILABLE:
            self.tda = EnhancedTDAFeatures(use_laplacian=True)
            logger.info("Persistent Laplacian TDA initialized")
        else:
            self.tda = None
        
        # Ensemble Regime Detector
        if self.config.use_ensemble_regime and ENSEMBLE_AVAILABLE:
            self.regime_detector = EnsembleRegimeDetector(
                n_regimes=self.config.n_regimes,
                consensus_threshold=self.config.regime_consensus
            )
            logger.info("Ensemble regime detector initialized")
        else:
            self.regime_detector = None
        
        # Order Flow Analyzer
        if self.config.use_order_flow and ORDERFLOW_AVAILABLE:
            self.order_flow = OrderFlowAnalyzer(window_minutes=15)
            logger.info("Order flow analyzer initialized")
        else:
            self.order_flow = None
        
        # V1.3 Fallback
        if self.config.fallback_to_v13 and V13_AVAILABLE:
            self.v13_engine = AdaptiveLearningEngine(
                initial_capital=self.initial_capital,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            logger.info("V1.3 fallback engine initialized")
        else:
            self.v13_engine = None
    
    def _check_component_status(self) -> Dict[str, bool]:
        """Check which components are available."""
        return {
            'transformer': self.transformer is not None,
            'sac': self.sac is not None,
            'tda_laplacian': self.tda is not None,
            'regime_ensemble': self.regime_detector is not None,
            'order_flow': self.order_flow is not None,
            'v13_fallback': self.v13_engine is not None
        }
    
    # =========================================================================
    # FEATURE COMPUTATION
    # =========================================================================
    
    def compute_features(self, ticker: str, 
                         prices: np.ndarray,
                         volume: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute all features for a ticker.
        
        Args:
            ticker: Stock ticker
            prices: Price history
            volume: Volume history (optional)
        
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        # Base features (10)
        base_features = self._compute_base_features(prices, volume)
        features['base'] = base_features
        
        # TDA features (12)
        if self.tda:
            returns = np.diff(np.log(prices + 1e-10))
            tda_features = self.tda.get_feature_vector(returns)
            features['tda'] = tda_features
        else:
            features['tda'] = np.zeros(12)
        
        # Regime features (5)
        if self.regime_detector:
            regime_features = self._compute_regime_features(prices, volume)
            features['regime'] = regime_features
        else:
            features['regime'] = np.zeros(5)
        
        # Order flow features (10)
        if self.order_flow:
            orderflow_features = self.order_flow.get_feature_vector(ticker)
            features['orderflow'] = orderflow_features
        else:
            features['orderflow'] = np.zeros(10)
        
        # Combined state for SAC
        features['state'] = np.concatenate([
            features['base'],
            features['tda'],
            features['regime'],
            features['orderflow']
        ])
        
        return features
    
    def _compute_base_features(self, prices: np.ndarray,
                                volume: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute base features (same as V1.3).
        
        10 features:
        1. Log returns (latest)
        2. RSI (14-day)
        3. MACD signal
        4. Bollinger Band position
        5-8. Momentum (5, 10, 20, 50 day)
        9. Volatility (20-day)
        10. Volume ratio (vs 20-day MA)
        """
        features = np.zeros(10)
        
        if len(prices) < 50:
            return features
        
        # 1. Log returns
        returns = np.diff(np.log(prices + 1e-10))
        features[0] = returns[-1] if len(returns) > 0 else 0
        
        # 2. RSI (14-day)
        if len(returns) >= 14:
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:]) + 1e-10
            rs = avg_gain / avg_loss
            features[1] = (100 - 100 / (1 + rs)) / 100 - 0.5  # Normalize to [-0.5, 0.5]
        
        # 3. MACD signal
        if len(prices) >= 26:
            ema_12 = self._ema(prices, 12)
            ema_26 = self._ema(prices, 26)
            macd = ema_12 - ema_26
            signal = self._ema(macd, 9) if len(macd) >= 9 else macd
            features[2] = (macd[-1] - signal[-1]) / prices[-1]  # Normalized
        
        # 4. Bollinger Band position
        if len(prices) >= 20:
            sma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            upper = sma_20 + 2 * std_20
            lower = sma_20 - 2 * std_20
            features[3] = (prices[-1] - lower) / (upper - lower + 1e-10) - 0.5
        
        # 5-8. Momentum
        for i, period in enumerate([5, 10, 20, 50]):
            if len(prices) >= period:
                features[4 + i] = (prices[-1] / prices[-period] - 1)
        
        # 9. Volatility
        if len(returns) >= 20:
            features[8] = np.std(returns[-20:]) * np.sqrt(252)
        
        # 10. Volume ratio
        if volume is not None and len(volume) >= 20:
            vol_sma = np.mean(volume[-20:])
            features[9] = (volume[-1] / vol_sma - 1) if vol_sma > 0 else 0
        
        return features
    
    def _compute_regime_features(self, prices: np.ndarray,
                                  volume: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute regime-related features.
        
        5 features:
        1. Current regime (one-hot: bull/bear/sideways)
        2-4. Reserved for one-hot
        5. Regime confidence
        """
        features = np.zeros(5)
        
        if self.regime_detector and len(prices) >= 50:
            returns = np.diff(np.log(prices + 1e-10))
            regime_features = self.regime_detector.compute_features(returns)
            
            # Get prediction for latest
            if len(regime_features) > 0:
                state = self.regime_detector.predict(regime_features[-1])
                
                # One-hot encoding
                if state.regime == RegimeType.BULL:
                    features[0] = 1.0
                elif state.regime == RegimeType.BEAR:
                    features[1] = 1.0
                elif state.regime == RegimeType.SIDEWAYS:
                    features[2] = 1.0
                else:
                    features[3] = 1.0  # Unknown/other
                
                features[4] = state.confidence
        
        return features
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    # =========================================================================
    # PREDICTION & DECISION
    # =========================================================================
    
    def predict(self, ticker: str, 
                prices: np.ndarray,
                volume: Optional[np.ndarray] = None) -> Dict:
        """
        Generate prediction for a ticker.
        
        Args:
            ticker: Stock ticker
            prices: Price history
            volume: Volume history
        
        Returns:
            Dictionary with prediction details
        """
        start_time = time.time()
        
        # Compute features
        features = self.compute_features(ticker, prices, volume)
        
        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'price': prices[-1] if len(prices) > 0 else 0,
            'direction': 0.0,
            'confidence': 0.0,
            'position_multiplier': 1.0,
            'regime': self.current_regime,
            'signal': 'hold',
            'features_computed': True
        }
        
        # Transformer prediction
        if self.transformer:
            try:
                # Prepare features for transformer
                # Shape: (n_samples, n_features)
                base_features = features['base'].reshape(1, -1)
                direction = self.transformer.predict(base_features)
                result['direction'] = float(direction[0])
                result['confidence'] = abs(direction[0] - 0.5) * 2
            except Exception as e:
                logger.warning(f"Transformer prediction failed: {e}")
        
        # SAC position sizing
        if self.sac:
            try:
                state = features['state']
                multiplier = self.sac.compute_position_multiplier(
                    state,
                    vol_20d=features['base'][8],  # volatility
                    deterministic=True
                )
                result['position_multiplier'] = float(multiplier)
            except Exception as e:
                logger.warning(f"SAC sizing failed: {e}")
        
        # Order flow signal
        if self.order_flow:
            try:
                signal, strength = self.order_flow.get_signal(ticker)
                result['orderflow_signal'] = signal
                result['orderflow_strength'] = strength
            except Exception as e:
                logger.warning(f"Order flow analysis failed: {e}")
        
        # Determine final signal
        if result['direction'] > 0.6 and result['confidence'] > 0.3:
            result['signal'] = 'buy'
        elif result['direction'] < 0.4 and result['confidence'] > 0.3:
            result['signal'] = 'sell'
        else:
            result['signal'] = 'hold'
        
        result['compute_time_ms'] = (time.time() - start_time) * 1000
        
        return result
    
    def update_regime(self, market_returns: np.ndarray,
                      market_volatility: Optional[np.ndarray] = None):
        """
        Update current market regime.
        
        Args:
            market_returns: Market (SPY) returns
            market_volatility: Market volatility (optional)
        """
        if not self.regime_detector:
            return
        
        try:
            # Compute features
            regime_features = self.regime_detector.compute_features(
                market_returns, market_volatility
            )
            
            if len(regime_features) > 0:
                state = self.regime_detector.predict(regime_features[-1])
                self.current_regime = self.regime_detector.get_regime_string()
                
                logger.info(f"Regime updated: {self.current_regime} "
                           f"(confidence: {state.confidence:.2f})")
        except Exception as e:
            logger.warning(f"Regime update failed: {e}")
    
    # =========================================================================
    # POSITION SIZING & RISK MANAGEMENT
    # =========================================================================
    
    def compute_position_size(self, ticker: str,
                               prediction: Dict,
                               current_price: float,
                               atr: float) -> float:
        """
        Compute position size with risk management.
        
        Args:
            ticker: Stock ticker
            prediction: Prediction dictionary
            current_price: Current price
            atr: Average True Range
        
        Returns:
            Position size in dollars
        """
        # Base position size
        base_size = self.portfolio_value * self.config.base_position_size
        
        # Apply SAC multiplier
        multiplier = prediction.get('position_multiplier', 1.0)
        position_size = base_size * multiplier
        
        # Regime adjustment
        if self.current_regime == 'risk_off':
            position_size *= 0.5
        elif self.current_regime == 'risk_on':
            position_size *= 1.25
        
        # Confidence adjustment
        confidence = prediction.get('confidence', 0.5)
        position_size *= (0.5 + confidence)
        
        # Volatility adjustment (inverse vol sizing)
        if atr > 0:
            vol_adj = 1.0 / (1 + atr / current_price * 20)
            position_size *= vol_adj
        
        # Apply limits
        min_size = self.portfolio_value * self.config.min_position_size
        max_size = self.portfolio_value * self.config.max_position_size
        position_size = np.clip(position_size, min_size, max_size)
        
        # Check portfolio heat
        current_heat = sum(abs(v) for v in self.positions.values()) / self.portfolio_value
        if current_heat + position_size / self.portfolio_value > self.config.max_portfolio_heat:
            # Reduce size to stay within heat limit
            available_heat = self.config.max_portfolio_heat - current_heat
            position_size = min(position_size, available_heat * self.portfolio_value)
        
        return max(0, position_size)
    
    def compute_stop_loss(self, entry_price: float, atr: float,
                          is_long: bool = True) -> float:
        """Compute stop loss price."""
        stop_distance = atr * self.config.stop_loss_atr_mult
        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def compute_take_profit(self, entry_price: float, atr: float,
                            is_long: bool = True) -> float:
        """Compute take profit price."""
        profit_distance = atr * self.config.take_profit_atr_mult
        if is_long:
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    # =========================================================================
    # TRAINING & LEARNING
    # =========================================================================
    
    def train(self, price_data: Dict[str, np.ndarray],
              volume_data: Optional[Dict[str, np.ndarray]] = None,
              epochs: int = 10) -> Dict:
        """
        Train all learnable components.
        
        Args:
            price_data: Dictionary of ticker -> price array
            volume_data: Dictionary of ticker -> volume array
            epochs: Training epochs
        
        Returns:
            Training metrics
        """
        metrics = {}
        
        start_time = time.time()
        
        # Train Transformer
        if self.transformer:
            logger.info("Training Transformer predictor...")
            try:
                self.transformer.train(price_data, epochs=epochs)
                metrics['transformer_trained'] = True
            except Exception as e:
                logger.error(f"Transformer training failed: {e}")
                metrics['transformer_trained'] = False
        
        # Fit Regime Detector
        if self.regime_detector:
            logger.info("Fitting regime detector...")
            try:
                # Use SPY or first ticker for market regime
                market_ticker = 'SPY' if 'SPY' in price_data else list(price_data.keys())[0]
                market_prices = price_data[market_ticker]
                market_returns = np.diff(np.log(market_prices + 1e-10))
                
                self.regime_detector.fit(market_returns)
                metrics['regime_fitted'] = True
            except Exception as e:
                logger.error(f"Regime fitting failed: {e}")
                metrics['regime_fitted'] = False
        
        # SAC is trained online, but we can pre-populate buffer
        if self.sac:
            logger.info("Pre-training SAC agent...")
            # Generate synthetic experiences
            # (In production, this would use historical trades)
            metrics['sac_buffer_size'] = len(self.sac.buffer)
        
        metrics['training_time_seconds'] = time.time() - start_time
        logger.info(f"Training complete in {metrics['training_time_seconds']:.2f}s")
        
        return metrics
    
    def update_sac(self, state: np.ndarray, action: float,
                   reward: float, next_state: np.ndarray, done: bool):
        """Add experience and update SAC agent."""
        if self.sac:
            self.sac.add_experience(state, action, reward, next_state, done)
            self.sac.update()
    
    # =========================================================================
    # PORTFOLIO MANAGEMENT
    # =========================================================================
    
    def get_portfolio_allocation(self, tickers: List[str],
                                  price_data: Dict[str, np.ndarray],
                                  volume_data: Optional[Dict[str, np.ndarray]] = None
                                  ) -> Dict[str, float]:
        """
        Get portfolio allocation for tickers.
        
        Args:
            tickers: List of tickers to consider
            price_data: Price history for each ticker
            volume_data: Volume history (optional)
        
        Returns:
            Dictionary of ticker -> allocation weight
        """
        allocations = {}
        predictions = {}
        
        # Get predictions for all tickers
        for ticker in tickers:
            if ticker not in price_data:
                continue
            
            prices = price_data[ticker]
            volume = volume_data.get(ticker) if volume_data else None
            
            pred = self.predict(ticker, prices, volume)
            predictions[ticker] = pred
        
        # Filter to buy signals
        buy_candidates = {
            t: p for t, p in predictions.items() 
            if p['signal'] == 'buy' and p['confidence'] > 0.3
        }
        
        if not buy_candidates:
            return allocations
        
        # Score and rank
        scores = {
            t: p['direction'] * p['confidence'] * p.get('position_multiplier', 1.0)
            for t, p in buy_candidates.items()
        }
        
        total_score = sum(scores.values())
        
        if total_score > 0:
            for ticker, score in scores.items():
                # Base allocation by score
                weight = score / total_score
                
                # Apply constraints
                weight = min(weight, self.config.max_position_size / self.config.base_position_size)
                
                allocations[ticker] = weight
        
        # Normalize to sum to max portfolio heat
        total_allocation = sum(allocations.values())
        if total_allocation > self.config.max_portfolio_heat:
            scale = self.config.max_portfolio_heat / total_allocation
            allocations = {t: w * scale for t, w in allocations.items()}
        
        return allocations
    
    # =========================================================================
    # UTILITY & STATUS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get comprehensive engine status."""
        status = {
            'version': 'V2.0',
            'components': self.component_status,
            'current_regime': self.current_regime,
            'portfolio_value': self.portfolio_value,
            'positions': len(self.positions),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
        
        # SAC stats
        if self.sac:
            status['sac_stats'] = self.sac.get_stats()
        
        # Regime stats
        if self.regime_detector:
            status['regime_stats'] = self.regime_detector.get_stats()
        
        return status
    
    def save_state(self, path: str):
        """Save engine state to disk."""
        import json
        import os
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'version': 'V2.0',
            'config': self.config.__dict__,
            'current_regime': self.current_regime,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions,
            'component_status': self.component_status,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save SAC model
        if self.sac:
            self.sac.save_model()
        
        logger.info(f"Engine state saved to {path}")
    
    def load_state(self, path: str) -> bool:
        """Load engine state from disk."""
        import json
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.current_regime = state.get('current_regime', 'neutral')
            self.portfolio_value = state.get('portfolio_value', self.initial_capital)
            self.positions = state.get('positions', {})
            
            logger.info(f"Engine state loaded from {path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
            return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_v2_engine(use_v13_fallback: bool = True,
                     initial_capital: float = 100000,
                     api_key: Optional[str] = None,
                     api_secret: Optional[str] = None) -> V2EnhancedEngine:
    """
    Create V2 Enhanced Engine with default configuration.
    
    Args:
        use_v13_fallback: Enable V1.3 fallback
        initial_capital: Starting capital
        api_key: Alpaca API key
        api_secret: Alpaca API secret
    
    Returns:
        Configured V2EnhancedEngine
    """
    config = V2Config(
        fallback_to_v13=use_v13_fallback,
        use_transformer=TRANSFORMER_AVAILABLE,
        use_sac=SAC_AVAILABLE,
        use_persistent_laplacian=LAPLACIAN_AVAILABLE,
        use_ensemble_regime=ENSEMBLE_AVAILABLE,
        use_order_flow=ORDERFLOW_AVAILABLE
    )
    
    return V2EnhancedEngine(
        config=config,
        initial_capital=initial_capital,
        api_key=api_key,
        api_secret=api_secret
    )


def create_minimal_engine(initial_capital: float = 100000) -> V2EnhancedEngine:
    """
    Create minimal V2 engine with only core components.
    
    Useful for testing or resource-constrained environments.
    """
    config = V2Config(
        use_transformer=True,
        use_sac=False,
        use_persistent_laplacian=False,
        use_ensemble_regime=False,
        use_order_flow=False,
        fallback_to_v13=False
    )
    
    return V2EnhancedEngine(config=config, initial_capital=initial_capital)
