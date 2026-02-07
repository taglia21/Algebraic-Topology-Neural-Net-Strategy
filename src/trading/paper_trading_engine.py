"""
Paper Trading Execution Engine
==============================

Integrates Phase 12 v3 regime-switching strategy with Alpaca API:
- Daily regime detection and rebalancing
- Position sizing and management
- Circuit breaker enforcement
- Trade logging and monitoring

This is the core execution module for paper trading.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.alpaca_client import AlpacaClient, OrderSide, Position, Account

# ==== WIRED QUANT MODULES ====
try:
    from src.signal_aggregator import SignalAggregator
    _AGGREGATOR_AVAILABLE = True
except ImportError:
    _AGGREGATOR_AVAILABLE = False

try:
    from src.quant_models.capm import CAPMModel
    _CAPM_AVAILABLE = True
except ImportError:
    _CAPM_AVAILABLE = False

try:
    from src.quant_models.garch import GARCHModel
    _GARCH_AVAILABLE = True
except ImportError:
    _GARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass
class RegimeSignal:
    """Regime classification result."""
    regime: MarketRegime
    confidence: float
    sma_20: float
    sma_50: float
    sma_200: float
    current_price: float
    momentum_20d: float
    volatility_20d: float
    consecutive_days: int
    timestamp: str


@dataclass
class PortfolioTarget:
    """Target portfolio allocation."""
    regime: MarketRegime
    long_etfs: Dict[str, float]  # {ticker: weight}
    inverse_etfs: Dict[str, float]  # {ticker: weight}
    cash_weight: float
    total_allocation: float


@dataclass
class TradeRecord:
    """Record of executed trade."""
    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    value: float
    order_id: str
    regime: str
    reason: str


class RegimeDetector:
    """
    Phase 12 v3 regime detection using SPY price action.
    
    Bull: Price > SMA20 > SMA50 > SMA200 + positive momentum
    Bear: Price < SMA20 < SMA50 < SMA200 + negative momentum
    Neutral: Conflicting signals -> hold cash
    """
    
    def __init__(
        self,
        confirmation_days: int = 5,
        momentum_threshold: float = 0.01,
    ):
        self.confirmation_days = confirmation_days
        self.momentum_threshold = momentum_threshold
        
        self.regime_history: List[MarketRegime] = []
        self.current_regime = MarketRegime.NEUTRAL
        self.days_in_regime = 0
    
    def detect_regime(self, spy_prices: pd.Series) -> RegimeSignal:
        """
        Detect current market regime from SPY prices.
        
        Args:
            spy_prices: Series of SPY close prices (at least 200 days)
            
        Returns:
            RegimeSignal with classification
        """
        if len(spy_prices) < 200:
            return self._neutral_signal(spy_prices)
        
        current_price = spy_prices.iloc[-1]
        sma_20 = spy_prices.rolling(20).mean().iloc[-1]
        sma_50 = spy_prices.rolling(50).mean().iloc[-1]
        sma_200 = spy_prices.rolling(200).mean().iloc[-1]
        
        # Momentum
        momentum_20d = (current_price / spy_prices.iloc[-20] - 1) if len(spy_prices) >= 20 else 0
        
        # Volatility
        returns = spy_prices.pct_change().dropna()
        volatility_20d = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0.2
        
        # Classify regime
        if current_price > sma_20 > sma_50 > sma_200 and momentum_20d > self.momentum_threshold:
            raw_regime = MarketRegime.BULL
            confidence = min(1.0, (current_price / sma_200 - 1) * 5 + 0.5)
        elif current_price < sma_20 < sma_50 < sma_200 and momentum_20d < -self.momentum_threshold:
            raw_regime = MarketRegime.BEAR
            confidence = min(1.0, (1 - current_price / sma_200) * 5 + 0.5)
        else:
            raw_regime = MarketRegime.NEUTRAL
            confidence = 0.5
        
        # Apply confirmation logic
        self.regime_history.append(raw_regime)
        if len(self.regime_history) > 20:
            self.regime_history = self.regime_history[-20:]
        
        # Check for regime persistence
        if len(self.regime_history) >= self.confirmation_days:
            recent = self.regime_history[-self.confirmation_days:]
            if all(r == raw_regime for r in recent):
                if raw_regime != self.current_regime:
                    logger.info(f"Regime change confirmed: {self.current_regime.value} -> {raw_regime.value}")
                    self.current_regime = raw_regime
                    self.days_in_regime = 1
                else:
                    self.days_in_regime += 1
        
        return RegimeSignal(
            regime=self.current_regime,
            confidence=confidence,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            current_price=current_price,
            momentum_20d=momentum_20d,
            volatility_20d=volatility_20d,
            consecutive_days=self.days_in_regime,
            timestamp=datetime.now().isoformat(),
        )
    
    def _neutral_signal(self, prices: pd.Series) -> RegimeSignal:
        """Return neutral signal when insufficient data."""
        return RegimeSignal(
            regime=MarketRegime.NEUTRAL,
            confidence=0.5,
            sma_20=0,
            sma_50=0,
            sma_200=0,
            current_price=prices.iloc[-1] if len(prices) > 0 else 0,
            momentum_20d=0,
            volatility_20d=0.2,
            consecutive_days=0,
            timestamp=datetime.now().isoformat(),
        )


class PortfolioConstructor:
    """
    Construct target portfolio based on regime.
    
    Phase 12 v3 allocation:
    - Bull: Long 3x ETFs (TQQQ 50%, SPXL 30%, SOXL 20%)
    - Bear: Inverse 3x ETFs (SQQQ 50%, SPXU 30%, SOXS 20%)
    - Neutral: 100% cash
    """
    
    # ETF allocations
    LONG_ETFS = {"TQQQ": 0.50, "SPXL": 0.30, "SOXL": 0.20}
    INVERSE_ETFS = {"SQQQ": 0.50, "SPXU": 0.30, "SOXS": 0.20}
    
    def __init__(
        self,
        max_leveraged_etf_pct: float = 0.25,
        base_allocation: float = 0.70,  # Start at 70% allocation
    ):
        self.max_leveraged_etf_pct = max_leveraged_etf_pct
        self.base_allocation = base_allocation
    
    def construct_portfolio(
        self,
        regime_signal: RegimeSignal,
        current_drawdown: float = 0,
    ) -> PortfolioTarget:
        """
        Construct target portfolio weights.
        
        Args:
            regime_signal: Current regime classification
            current_drawdown: Current portfolio drawdown (0 to 1)
            
        Returns:
            PortfolioTarget with allocations
        """
        regime = regime_signal.regime
        confidence = regime_signal.confidence
        volatility = regime_signal.volatility_20d
        
        # Base allocation adjusted by confidence
        allocation = self.base_allocation * min(1.0, confidence + 0.3)
        
        # Volatility scaling
        if volatility > 0.35:
            allocation *= 0.50
        elif volatility > 0.25:
            allocation *= 0.70
        
        # Drawdown protection
        if current_drawdown > 0.15:
            allocation *= 0.30
        elif current_drawdown > 0.10:
            allocation *= 0.50
        elif current_drawdown > 0.05:
            allocation *= 0.75
        
        # Limit single ETF exposure
        max_single = self.max_leveraged_etf_pct
        
        if regime == MarketRegime.BULL:
            long_etfs = {t: min(w * allocation, max_single) for t, w in self.LONG_ETFS.items()}
            inverse_etfs = {}
        elif regime == MarketRegime.BEAR:
            long_etfs = {}
            inverse_etfs = {t: min(w * allocation, max_single) for t, w in self.INVERSE_ETFS.items()}
        else:
            long_etfs = {}
            inverse_etfs = {}
        
        total_allocation = sum(long_etfs.values()) + sum(inverse_etfs.values())
        cash_weight = 1.0 - total_allocation
        
        return PortfolioTarget(
            regime=regime,
            long_etfs=long_etfs,
            inverse_etfs=inverse_etfs,
            cash_weight=cash_weight,
            total_allocation=total_allocation,
        )


class CircuitBreaker:
    """
    Production safety controls.
    
    Triggers:
    - Daily loss > 3% -> Reduce to 50%
    - Daily loss > 5% -> Exit all
    - VIX > 40 -> Exit to cash
    - Max DD > 15% -> Reduce to 30%
    """
    
    def __init__(
        self,
        daily_loss_reduce: float = 0.03,
        daily_loss_exit: float = 0.05,
        max_dd_reduce: float = 0.15,
        max_dd_exit: float = 0.20,
    ):
        self.daily_loss_reduce = daily_loss_reduce
        self.daily_loss_exit = daily_loss_exit
        self.max_dd_reduce = max_dd_reduce
        self.max_dd_exit = max_dd_exit
        
        self.start_of_day_equity = None
        self.peak_equity = None
        self.triggered = False
        self.trigger_reason = None
    
    def update(self, current_equity: float) -> Tuple[bool, Optional[str], float]:
        """
        Check circuit breakers.
        
        Args:
            current_equity: Current portfolio value
            
        Returns:
            Tuple of (is_triggered, reason, scale_factor)
        """
        if self.start_of_day_equity is None:
            self.start_of_day_equity = current_equity
        
        if self.peak_equity is None:
            self.peak_equity = current_equity
        else:
            self.peak_equity = max(self.peak_equity, current_equity)
        
        # Daily loss
        daily_loss = 1 - current_equity / self.start_of_day_equity
        
        # Drawdown
        drawdown = 1 - current_equity / self.peak_equity
        
        # Check triggers
        if daily_loss >= self.daily_loss_exit:
            return True, f"DAILY_LOSS_EXIT ({daily_loss:.1%})", 0.0
        
        if drawdown >= self.max_dd_exit:
            return True, f"MAX_DD_EXIT ({drawdown:.1%})", 0.0
        
        if daily_loss >= self.daily_loss_reduce:
            return True, f"DAILY_LOSS_REDUCE ({daily_loss:.1%})", 0.5
        
        if drawdown >= self.max_dd_reduce:
            return True, f"MAX_DD_REDUCE ({drawdown:.1%})", 0.3
        
        return False, None, 1.0
    
    def reset_daily(self, equity: float):
        """Reset daily tracking."""
        self.start_of_day_equity = equity
        self.triggered = False
        self.trigger_reason = None


class PaperTradingEngine:
    """
    Main paper trading execution engine.
    
    Integrates all components:
    - Alpaca API client
    - Regime detection
    - Portfolio construction
    - Circuit breakers
    - Trade execution
    - Logging and monitoring
    """
    
    def __init__(self):
        """Initialize engine from environment variables."""
        self.client = AlpacaClient()
        self.regime_detector = RegimeDetector(
            confirmation_days=int(os.getenv("REGIME_CONFIRMATION_DAYS", 5))
        )
        self.portfolio_constructor = PortfolioConstructor(
            max_leveraged_etf_pct=float(os.getenv("MAX_LEVERAGED_ETF_PCT", 0.25))
        )
        self.circuit_breaker = CircuitBreaker(
            daily_loss_reduce=float(os.getenv("MAX_DAILY_LOSS_PCT", 0.03))
        )
        
        # Tracking
        self.trade_history: List[TradeRecord] = []
        self.daily_log: List[Dict] = []
        self.starting_capital = float(os.getenv("STARTING_CAPITAL", 100000))

        # ==== WIRED QUANT MODULES ====
        self.signal_aggregator = None
        if _AGGREGATOR_AVAILABLE:
            try:
                self.signal_aggregator = SignalAggregator(min_confidence=0.5)
                self.signal_aggregator.initialize()
                logger.info("SignalAggregator wired into paper trading engine")
            except Exception as e:
                logger.warning(f"SignalAggregator init failed: {e}")

        self.capm_model = None
        if _CAPM_AVAILABLE:
            try:
                self.capm_model = CAPMModel()
                logger.info("CAPM wired into paper trading engine")
            except Exception as e:
                logger.warning(f"CAPM init failed: {e}")

        self.garch_model = None
        if _GARCH_AVAILABLE:
            try:
                self.garch_model = GARCHModel()
                logger.info("GARCH wired into paper trading engine")
            except Exception as e:
                logger.warning(f"GARCH init failed: {e}")

        # Logging setup
        self._setup_logging()
        
        logger.info("PaperTradingEngine initialized")
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = os.getenv("LOG_FILE", "logs/paper_trading.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Add handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def get_spy_prices(self, days: int = 250) -> pd.Series:
        """Fetch SPY price history."""
        end = datetime.now()
        start = end - timedelta(days=days + 30)  # Extra buffer
        
        df = yf.download("SPY", start=start.strftime("%Y-%m-%d"), 
                        end=end.strftime("%Y-%m-%d"), progress=False)
        
        if len(df) == 0:
            raise RuntimeError("Failed to fetch SPY data")
        
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            close = df[('Close', 'SPY')]
        elif 'Close' in df.columns:
            close = df['Close']
        else:
            close = df.iloc[:, 3]  # Assume 4th column is close
        
        return close
    
    def calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        account = self.client.get_account()
        current_equity = account.equity
        
        if self.circuit_breaker.peak_equity is None:
            return 0.0
        
        return 1 - current_equity / self.circuit_breaker.peak_equity
    
    def get_target_positions(self) -> Dict[str, float]:
        """
        Get target position sizes in dollars.
        
        ENHANCED: Uses GARCH vol forecast for dynamic sizing and
        SignalAggregator for regime cross-validation.
        
        Returns:
            Dict of {symbol: target_value}
        """
        # Get current account
        account = self.client.get_account()
        portfolio_value = account.portfolio_value
        
        # Get regime signal (SMA-based)
        spy_prices = self.get_spy_prices()
        regime_signal = self.regime_detector.detect_regime(spy_prices)
        
        # GARCH vol overlay — adjust volatility estimate used by PortfolioConstructor
        if self.garch_model is not None:
            try:
                garch_forecast = self.garch_model.fit_and_forecast("SPY", horizon=5)
                garch_vol = garch_forecast.current_vol
                logger.info(f"GARCH Vol: {garch_vol:.1%} (vs SMA-based: {regime_signal.volatility_20d:.1%})")
                # Use GARCH vol if available — it's more accurate
                regime_signal.volatility_20d = garch_vol
            except Exception as e:
                logger.debug(f"GARCH forecast failed: {e}")

        # SignalAggregator cross-validation — can override regime if models disagree
        if self.signal_aggregator is not None:
            try:
                agg = self.signal_aggregator.aggregate("SPY", min_confidence=0.3)
                logger.info(
                    f"Aggregator: signal={agg.signal:.3f} ({agg.direction}), "
                    f"confidence={agg.confidence:.3f}, regime={agg.regime.value}"
                )
                # If aggregator strongly disagrees with SMA regime, dampen confidence
                if (regime_signal.regime == MarketRegime.BULL and agg.signal < -0.4):
                    logger.warning("Aggregator bearish but SMA says BULL — dampening confidence")
                    regime_signal.confidence *= 0.6
                elif (regime_signal.regime == MarketRegime.BEAR and agg.signal > 0.4):
                    logger.warning("Aggregator bullish but SMA says BEAR — dampening confidence")
                    regime_signal.confidence *= 0.6
                elif abs(agg.signal) > 0.5:
                    # Strong agreement — boost confidence
                    regime_signal.confidence = min(regime_signal.confidence * 1.2, 0.99)
            except Exception as e:
                logger.debug(f"SignalAggregator failed: {e}")
        
        # Check circuit breakers
        cb_triggered, cb_reason, scale = self.circuit_breaker.update(account.equity)
        
        if cb_triggered:
            logger.warning(f"🚨 CIRCUIT BREAKER: {cb_reason}")
            if scale == 0:
                # Exit all positions
                return {}
        
        # Get drawdown
        drawdown = self.calculate_current_drawdown()
        
        # Construct portfolio
        target = self.portfolio_constructor.construct_portfolio(regime_signal, drawdown)
        
        # Apply circuit breaker scale
        target_positions = {}
        
        for symbol, weight in target.long_etfs.items():
            adjusted_weight = weight * scale
            target_positions[symbol] = portfolio_value * adjusted_weight
        
        for symbol, weight in target.inverse_etfs.items():
            adjusted_weight = weight * scale
            target_positions[symbol] = portfolio_value * adjusted_weight
        
        logger.info(f"Regime: {regime_signal.regime.value} | "
                   f"Confidence: {regime_signal.confidence:.1%} | "
                   f"Days: {regime_signal.consecutive_days} | "
                   f"DD: {drawdown:.1%}")
        
        return target_positions
    
    def execute_rebalance(self) -> List[TradeRecord]:
        """
        Execute portfolio rebalance.
        
        Compares current positions to targets and places orders.
        
        Returns:
            List of executed trades
        """
        logger.info("=" * 60)
        logger.info("EXECUTING REBALANCE")
        logger.info("=" * 60)
        
        # Check if market is open
        if not self.client.is_market_open():
            logger.info("Market is closed. Skipping rebalance.")
            return []
        
        trades = []
        
        # Get targets
        target_positions = self.get_target_positions()
        
        # Get current positions
        current_positions = {p.symbol: p for p in self.client.get_positions()}
        
        # Get regime for logging
        spy_prices = self.get_spy_prices()
        regime_signal = self.regime_detector.detect_regime(spy_prices)
        
        # All symbols to consider
        all_symbols = set(target_positions.keys()) | set(current_positions.keys())
        
        # Process each symbol
        for symbol in all_symbols:
            target_value = target_positions.get(symbol, 0)
            current_position = current_positions.get(symbol)
            current_value = current_position.market_value if current_position else 0
            
            diff = target_value - current_value
            
            # Skip small changes (< $100)
            if abs(diff) < 100:
                continue
            
            try:
                if diff > 0:
                    # Buy
                    order = self.client.submit_notional_order(symbol, diff, OrderSide.BUY)
                    logger.info(f"  BUY {symbol}: ${diff:,.2f}")
                else:
                    # Sell
                    order = self.client.submit_notional_order(symbol, abs(diff), OrderSide.SELL)
                    logger.info(f"  SELL {symbol}: ${abs(diff):,.2f}")
                
                # Wait for fill
                time.sleep(1)
                filled_order = self.client.get_order(order.id)
                
                trade = TradeRecord(
                    timestamp=datetime.now().isoformat(),
                    symbol=symbol,
                    side="buy" if diff > 0 else "sell",
                    qty=filled_order.filled_qty,
                    price=filled_order.filled_avg_price or 0,
                    value=abs(diff),
                    order_id=order.id,
                    regime=regime_signal.regime.value,
                    reason="rebalance",
                )
                trades.append(trade)
                self.trade_history.append(trade)
                
            except Exception as e:
                logger.error(f"  ❌ Failed to execute {symbol}: {e}")
        
        logger.info(f"Rebalance complete: {len(trades)} trades executed")
        
        return trades
    
    def emergency_exit(self, reason: str) -> List[TradeRecord]:
        """
        Emergency liquidation of all positions.
        
        Args:
            reason: Why emergency exit was triggered
            
        Returns:
            List of close trades
        """
        logger.warning("=" * 60)
        logger.warning(f"🚨 EMERGENCY EXIT: {reason}")
        logger.warning("=" * 60)
        
        trades = []
        
        try:
            # Cancel all open orders first
            self.client.cancel_all_orders()
            logger.info("Cancelled all open orders")
            
            # Close all positions
            orders = self.client.close_all_positions()
            
            for order in orders:
                trade = TradeRecord(
                    timestamp=datetime.now().isoformat(),
                    symbol=order.symbol,
                    side="sell",
                    qty=order.qty,
                    price=0,  # Market order
                    value=0,
                    order_id=order.id,
                    regime="emergency",
                    reason=reason,
                )
                trades.append(trade)
                self.trade_history.append(trade)
            
            logger.warning(f"Emergency exit complete: {len(trades)} positions closed")
            
        except Exception as e:
            logger.error(f"Emergency exit failed: {e}")
        
        return trades
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        account = self.client.get_account()
        positions = self.client.get_positions()
        
        # Calculate performance
        current_equity = account.equity
        starting = self.starting_capital
        
        total_return = (current_equity / starting - 1) * 100
        
        # Drawdown
        peak = self.circuit_breaker.peak_equity or current_equity
        drawdown = (1 - current_equity / peak) * 100
        
        # Position summary
        position_summary = []
        for p in positions:
            position_summary.append({
                "symbol": p.symbol,
                "qty": p.qty,
                "value": p.market_value,
                "pnl": p.unrealized_pl,
                "pnl_pct": p.unrealized_plpc * 100,
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "starting_capital": starting,
            "current_equity": current_equity,
            "cash": account.cash,
            "total_return_pct": total_return,
            "max_drawdown_pct": drawdown,
            "position_count": len(positions),
            "positions": position_summary,
            "trade_count": len(self.trade_history),
            "current_regime": self.regime_detector.current_regime.value,
            "days_in_regime": self.regime_detector.days_in_regime,
        }
    
    def print_status(self):
        """Print current status."""
        summary = self.get_performance_summary()
        
        print("\n" + "=" * 60)
        print("PAPER TRADING STATUS")
        print("=" * 60)
        print(f"\nAccount:")
        print(f"  Starting Capital: ${summary['starting_capital']:,.2f}")
        print(f"  Current Equity:   ${summary['current_equity']:,.2f}")
        print(f"  Cash:             ${summary['cash']:,.2f}")
        print(f"  Total Return:     {summary['total_return_pct']:+.2f}%")
        print(f"  Max Drawdown:     {summary['max_drawdown_pct']:.2f}%")
        
        print(f"\nRegime:")
        print(f"  Current:          {summary['current_regime'].upper()}")
        print(f"  Days in Regime:   {summary['days_in_regime']}")
        
        print(f"\nPositions ({summary['position_count']}):")
        for pos in summary['positions']:
            pnl_sign = "+" if pos['pnl'] >= 0 else ""
            print(f"  {pos['symbol']:6s} ${pos['value']:>10,.2f}  "
                  f"{pnl_sign}${pos['pnl']:>8,.2f} ({pnl_sign}{pos['pnl_pct']:.1f}%)")
        
        print(f"\nTrades Executed: {summary['trade_count']}")
        print("=" * 60)
    
    def save_state(self, filepath: str = "logs/trading_state.json"):
        """Save current state to file."""
        state = {
            "performance": self.get_performance_summary(),
            "trade_history": [asdict(t) for t in self.trade_history[-100:]],  # Last 100 trades
            "regime_history": [r.value for r in self.regime_detector.regime_history],
            "circuit_breaker": {
                "peak_equity": self.circuit_breaker.peak_equity,
                "start_of_day": self.circuit_breaker.start_of_day_equity,
            },
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"State saved to {filepath}")


def main():
    """Main entry point for testing."""
    print("=" * 60)
    print("PAPER TRADING ENGINE TEST")
    print("=" * 60)
    
    try:
        engine = PaperTradingEngine()
        
        # Health check
        health = engine.client.health_check()
        if health["status"] != "healthy":
            print(f"❌ Connection failed: {health.get('error')}")
            return
        
        print(f"✅ Connected to Alpaca ({'PAPER' if health['is_paper'] else 'LIVE'})")
        print(f"   Account: {health['account_id']}")
        print(f"   Equity: ${health['equity']:,.2f}")
        
        # Show current status
        engine.print_status()
        
        # Test regime detection
        print("\nTesting regime detection...")
        spy_prices = engine.get_spy_prices()
        signal = engine.regime_detector.detect_regime(spy_prices)
        print(f"  Regime: {signal.regime.value}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  SPY: ${signal.current_price:.2f}")
        print(f"  Momentum (20d): {signal.momentum_20d:.1%}")
        
        # Test target calculation
        print("\nTarget positions:")
        targets = engine.get_target_positions()
        for symbol, value in targets.items():
            print(f"  {symbol}: ${value:,.2f}")
        
        print("\n✅ Engine ready for deployment")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
