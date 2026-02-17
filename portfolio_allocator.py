#!/usr/bin/env python3
"""
Portfolio Allocator — Capital Allocation & Risk Budgeting
==========================================================

Allocates capital across the three strategies (Pairs, Mean Reversion,
Momentum) while enforcing market-neutrality constraints:

  - 50% → Pairs Trading (market-neutral by construction)
  - 30% → Mean Reversion (dollar-neutral per signal)
  - 20% → Momentum (only active when A+B idle)

Risk controls:
  - Net market beta between -0.2 and +0.2
  - Max 5% per individual position
  - Max 15% per sector
  - Inverse-volatility weighting within each strategy
  - Quarterly Sharpe-based reallocation

Usage:
    from portfolio_allocator import PortfolioAllocator, AllocatorConfig

    allocator = PortfolioAllocator(AllocatorConfig())
    sized_signals = allocator.allocate(signals, equity, positions)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger("portfolio_allocator")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AllocatorConfig:
    """Configuration for portfolio allocation."""

    # Strategy capital allocation (must sum to 1.0)
    pairs_pct: float = 0.50        # 50% to pairs (market-neutral core)
    mr_pct: float = 0.30           # 30% to mean reversion
    momentum_pct: float = 0.20     # 20% to momentum (fallback only)

    # Position limits
    max_position_pct: float = 0.05     # 5% max per individual position
    max_sector_pct: float = 0.15       # 15% max per sector
    min_position_pct: float = 0.005    # 0.5% min position (below this, skip)

    # Market neutrality
    max_net_beta: float = 0.20     # Keep |net beta| < 0.2
    target_net_beta: float = 0.0   # Target perfectly market-neutral

    # Inverse-volatility weighting
    use_inv_vol_weighting: bool = True  # Weight positions by 1/volatility
    vol_lookback: int = 20              # 20-day realized vol for weighting

    # Dynamic reallocation
    reallocation_trades: int = 50       # Re-evaluate after 50 trades
    min_trades_for_sharpe: int = 10     # Need 10+ trades for Sharpe estimate
    sharpe_realloc_weight: float = 0.3  # 30% weight to Sharpe-based realloc

    # Cash reserve
    cash_reserve_pct: float = 0.05     # Keep 5% in cash always


# ============================================================================
# SECTOR MAPPING (mirrors unified_trader.py)
# ============================================================================

SECTOR_MAP = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "GOOGL": "technology", "META": "technology", "CRM": "technology",
    "AVGO": "technology", "AMD": "technology", "INTC": "technology",
    "CSCO": "technology", "ADBE": "technology", "ORCL": "technology",
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare",
    "ABBV": "healthcare", "PFE": "healthcare", "MRK": "healthcare",
    "TMO": "healthcare", "ABT": "healthcare",
    "JPM": "financials", "GS": "financials", "V": "financials",
    "MA": "financials", "BAC": "financials", "MS": "financials",
    "C": "financials", "WFC": "financials", "BLK": "financials",
    "SCHW": "financials",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "SLB": "energy", "EOG": "energy",
    "AMZN": "consumer", "TSLA": "consumer", "NFLX": "consumer",
    "HD": "consumer", "MCD": "consumer", "NKE": "consumer",
    "SBUX": "consumer", "TGT": "consumer",
    "KO": "staples", "PG": "staples", "COST": "staples",
    "WMT": "staples", "PEP": "staples", "CL": "staples",
    "CAT": "industrials", "HON": "industrials", "GE": "industrials",
    "DE": "industrials", "UPS": "industrials", "RTX": "industrials",
    "BA": "industrials", "LMT": "industrials",
    "AMT": "reits", "O": "reits", "PLD": "reits",
    "SPY": "etf", "QQQ": "etf", "IWM": "etf",
    "XLF": "etf", "XLE": "etf", "XLV": "etf",
}


def get_sector(symbol: str) -> str:
    """Get sector for a symbol."""
    return SECTOR_MAP.get(symbol, "unknown")


# ============================================================================
# PORTFOLIO ALLOCATOR
# ============================================================================

class PortfolioAllocator:
    """
    Allocates capital across strategies and enforces risk budgets.

    Flow:
      1. Compute strategy budgets (50/30/20 or Sharpe-based)
      2. For each signal, compute inverse-vol-weighted position size
      3. Enforce per-position, per-sector, and net-beta limits
      4. Return sized signals ready for execution
    """

    def __init__(self, config: AllocatorConfig = None):
        self.cfg = config or AllocatorConfig()

        # Track strategy performance for dynamic reallocation
        self._strategy_trades: Dict[str, List[float]] = {
            "pairs_trading": [],
            "mean_reversion": [],
            "momentum_regime": [],
        }
        self._total_trades = 0
        self._last_realloc_trade_count = 0

        # Current allocation weights (start at defaults)
        self._current_alloc = {
            "pairs_trading": self.cfg.pairs_pct,
            "mean_reversion": self.cfg.mr_pct,
            "momentum_regime": self.cfg.momentum_pct,
        }

    # ------------------------------------------------------------------
    # Main entry: allocate capital to signals
    # ------------------------------------------------------------------

    def allocate(
        self,
        signals: list,
        equity: float,
        current_positions: Optional[Dict[str, Any]] = None,
        volatilities: Optional[Dict[str, float]] = None,
    ) -> list:
        """
        Size all signals according to strategy budgets and risk limits.

        Parameters
        ----------
        signals : List[TradeSignal]
            Raw signals from strategy_engine.get_signals().
        equity : float
            Current portfolio equity.
        current_positions : dict, optional
            {symbol: {qty, entry_price, market_value, sector, ...}}
        volatilities : dict, optional
            {symbol: 20-day realized volatility} for inv-vol weighting.

        Returns
        -------
        List[TradeSignal]
            Signals with position_size_pct and shares filled in.
            Signals that violate risk limits are filtered out.
        """
        if current_positions is None:
            current_positions = {}
        if volatilities is None:
            volatilities = {}

        # Available capital = equity - cash reserve - existing positions
        existing_exposure = sum(
            abs(float(p.get("market_value", 0))) if isinstance(p, dict)
            else abs(float(getattr(p, "market_value", 0)))
            for p in current_positions.values()
        )
        available_capital = equity * (1.0 - self.cfg.cash_reserve_pct) - existing_exposure
        available_capital = max(0, available_capital)

        if available_capital < equity * 0.01:
            logger.info("Insufficient available capital for new positions")
            return []

        # Maybe rebalance strategy allocations based on realized Sharpe
        self._maybe_rebalance()

        # Compute capital budget per strategy
        budgets = self._compute_budgets(available_capital, current_positions)

        # Track remaining budget per strategy
        remaining: Dict[str, float] = dict(budgets)

        # Track sector exposure
        sector_exposure: Dict[str, float] = {}
        for sym, pos in current_positions.items():
            sector = get_sector(sym)
            val = abs(float(pos.get("market_value", 0)) if isinstance(pos, dict)
                      else float(getattr(pos, "market_value", 0)))
            sector_exposure[sector] = sector_exposure.get(sector, 0) + val

        # Track net beta (long exposure - short exposure)
        net_long = sum(
            float(pos.get("market_value", 0)) if isinstance(pos, dict)
            else float(getattr(pos, "market_value", 0))
            for pos in current_positions.values()
        )
        net_beta_estimate = net_long / equity if equity > 0 else 0.0

        sized_signals = []

        for sig in signals:
            strategy_key = sig.strategy.value

            # Skip CLOSE signals — they don't need sizing
            if sig.direction.value == "close":
                sized_signals.append(sig)
                continue

            # Check strategy budget
            budget = remaining.get(strategy_key, 0)
            if budget < equity * self.cfg.min_position_pct:
                logger.debug(
                    f"Skipping {sig.symbol}: {strategy_key} budget exhausted "
                    f"(${budget:.0f} remaining)"
                )
                continue

            # Compute position size
            raw_size_pct = sig.position_size_pct
            if raw_size_pct <= 0:
                raw_size_pct = self.cfg.max_position_pct * 0.5  # Default 2.5%

            # Apply inverse-volatility weighting
            if self.cfg.use_inv_vol_weighting and sig.symbol in volatilities:
                vol = volatilities[sig.symbol]
                if vol > 0:
                    # Higher vol -> smaller position
                    # Normalize: vol=0.01 (1%) -> scale=1.5, vol=0.03 (3%) -> scale=0.5
                    inv_vol_scale = min(2.0, max(0.25, 0.015 / vol))
                    raw_size_pct *= inv_vol_scale

            # Cap at max position size
            raw_size_pct = min(raw_size_pct, self.cfg.max_position_pct)
            raw_size_pct = max(raw_size_pct, self.cfg.min_position_pct)

            # Cap at strategy budget
            position_dollars = raw_size_pct * equity
            if position_dollars > budget:
                position_dollars = budget
                raw_size_pct = position_dollars / equity if equity > 0 else 0

            # Sector exposure check
            sector = get_sector(sig.symbol)
            current_sector_exp = sector_exposure.get(sector, 0)
            max_sector_dollars = equity * self.cfg.max_sector_pct
            if current_sector_exp + position_dollars > max_sector_dollars:
                # Reduce to fit within sector limit
                allowed = max_sector_dollars - current_sector_exp
                if allowed < equity * self.cfg.min_position_pct:
                    logger.debug(
                        f"Skipping {sig.symbol}: sector '{sector}' at limit "
                        f"({current_sector_exp / equity:.1%})"
                    )
                    continue
                position_dollars = allowed
                raw_size_pct = position_dollars / equity if equity > 0 else 0

            # Net beta check (for non-pairs trades)
            if sig.strategy.value != "pairs_trading":
                beta_impact = position_dollars / equity if equity > 0 else 0
                if sig.direction.value == "short":
                    beta_impact = -beta_impact

                projected_beta = net_beta_estimate + beta_impact
                if abs(projected_beta) > self.cfg.max_net_beta:
                    # Check if this signal actually REDUCES beta
                    if abs(projected_beta) > abs(net_beta_estimate):
                        logger.debug(
                            f"Skipping {sig.symbol}: would push net beta to "
                            f"{projected_beta:+.2f} (limit ±{self.cfg.max_net_beta})"
                        )
                        continue
                net_beta_estimate = projected_beta

            # Compute shares
            if sig.entry_price > 0:
                shares = int(position_dollars / sig.entry_price)
            else:
                shares = 0

            if shares <= 0:
                continue

            # Update signal
            sig.position_size_pct = raw_size_pct
            sig.shares = shares

            # Update tracking
            remaining[strategy_key] -= position_dollars
            sector_exposure[sector] = current_sector_exp + position_dollars

            sized_signals.append(sig)
            logger.debug(
                f"Sized {sig.direction.value} {sig.symbol}: "
                f"{shares} shares (${position_dollars:,.0f}, {raw_size_pct:.1%}) "
                f"via {strategy_key}"
            )

        logger.info(
            f"Portfolio allocator: {len(sized_signals)}/{len(signals)} signals sized "
            f"(net_beta={net_beta_estimate:+.2f})"
        )

        return sized_signals

    # ------------------------------------------------------------------
    # Strategy budget computation
    # ------------------------------------------------------------------

    def _compute_budgets(
        self,
        available_capital: float,
        positions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute dollar budget per strategy."""
        # Compute existing per-strategy exposure
        existing: Dict[str, float] = {
            "pairs_trading": 0, "mean_reversion": 0, "momentum_regime": 0,
        }
        for sym, pos in positions.items():
            strat = (pos.get("strategy", "unknown") if isinstance(pos, dict)
                     else getattr(pos, "strategy", "unknown"))
            if strat in existing:
                val = (abs(float(pos.get("market_value", 0))) if isinstance(pos, dict)
                       else abs(float(getattr(pos, "market_value", 0))))
                existing[strat] += val

        # Target allocation per strategy
        budgets = {}
        for strat, alloc_pct in self._current_alloc.items():
            target = available_capital * alloc_pct
            used = existing.get(strat, 0)
            budgets[strat] = max(0, target - used)

        logger.debug(
            f"Strategy budgets: "
            + ", ".join(f"{k}=${v:,.0f}" for k, v in budgets.items())
        )

        return budgets

    # ------------------------------------------------------------------
    # Dynamic reallocation based on realized Sharpe
    # ------------------------------------------------------------------

    def record_trade(self, strategy: str, pnl_pct: float):
        """Record a completed trade return for Sharpe tracking."""
        if strategy in self._strategy_trades:
            self._strategy_trades[strategy].append(pnl_pct)
            self._total_trades += 1

    def _maybe_rebalance(self):
        """
        Rebalance strategy allocations based on realized Sharpe ratios.

        Only adjusts after accumulating enough trades. Shifts allocation
        toward strategies with higher Sharpe, while keeping minimum
        allocations to maintain diversification.
        """
        if self._total_trades - self._last_realloc_trade_count < self.cfg.reallocation_trades:
            return

        sharpes = {}
        for strat, returns in self._strategy_trades.items():
            if len(returns) >= self.cfg.min_trades_for_sharpe:
                arr = np.array(returns)
                mean_r = np.mean(arr)
                std_r = np.std(arr)
                # Annualized Sharpe proxy (assuming ~250 trades/year)
                sharpes[strat] = (mean_r / std_r * np.sqrt(250)) if std_r > 1e-8 else 0.0
            else:
                sharpes[strat] = 0.0

        # If we have Sharpe data, blend with base allocation
        if any(v != 0 for v in sharpes.values()):
            # Softmax-like weighting: exp(sharpe) / sum(exp(sharpe))
            max_s = max(sharpes.values())
            exp_sharpes = {k: np.exp(v - max_s) for k, v in sharpes.items()}
            total_exp = sum(exp_sharpes.values())

            if total_exp > 0:
                sharpe_alloc = {k: v / total_exp for k, v in exp_sharpes.items()}

                # Blend: (1-w)*base + w*sharpe_alloc
                w = self.cfg.sharpe_realloc_weight
                base_alloc = {
                    "pairs_trading": self.cfg.pairs_pct,
                    "mean_reversion": self.cfg.mr_pct,
                    "momentum_regime": self.cfg.momentum_pct,
                }

                new_alloc = {}
                for strat in self._current_alloc:
                    blended = (1 - w) * base_alloc[strat] + w * sharpe_alloc.get(strat, 0)
                    # Enforce minimum 10% per strategy
                    new_alloc[strat] = max(0.10, blended)

                # Renormalize to sum to 1.0
                total = sum(new_alloc.values())
                if total > 0:
                    self._current_alloc = {k: v / total for k, v in new_alloc.items()}

                logger.info(
                    f"Rebalanced allocations — "
                    f"pairs={self._current_alloc['pairs_trading']:.0%}, "
                    f"mr={self._current_alloc['mean_reversion']:.0%}, "
                    f"mom={self._current_alloc['momentum_regime']:.0%} "
                    f"(Sharpes: {', '.join(f'{k}={v:.2f}' for k, v in sharpes.items())})"
                )

        self._last_realloc_trade_count = self._total_trades

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_current_allocation(self) -> Dict[str, float]:
        """Get current strategy allocation percentages."""
        return dict(self._current_alloc)

    def get_net_beta_estimate(
        self,
        positions: Dict[str, Any],
        equity: float,
    ) -> float:
        """Estimate portfolio net beta from current positions."""
        if equity <= 0:
            return 0.0
        net = 0.0
        for sym, pos in positions.items():
            val = (float(pos.get("market_value", 0)) if isinstance(pos, dict)
                   else float(getattr(pos, "market_value", 0)))
            net += val  # Long is positive, short is negative
        return net / equity

    def get_strategy_stats(self) -> Dict[str, dict]:
        """Get per-strategy trade stats."""
        stats = {}
        for strat, returns in self._strategy_trades.items():
            arr = np.array(returns) if returns else np.array([0.0])
            stats[strat] = {
                "trades": len(returns),
                "win_rate": float(np.mean(arr > 0)) if len(returns) > 0 else 0.0,
                "avg_return": float(np.mean(arr)) if len(returns) > 0 else 0.0,
                "sharpe": float(np.mean(arr) / np.std(arr) * np.sqrt(250))
                         if len(returns) > 1 and np.std(arr) > 1e-8 else 0.0,
                "allocation": self._current_alloc.get(strat, 0.0),
            }
        return stats


# ============================================================================
# MAIN — Standalone test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    print("=" * 60)
    print("PORTFOLIO ALLOCATOR — Capital Allocation Test")
    print("=" * 60)

    allocator = PortfolioAllocator(AllocatorConfig())

    # Simulate some trade signals
    from strategy_engine import TradeSignal, SignalDirection, StrategyType

    test_signals = [
        TradeSignal(
            symbol="AAPL", direction=SignalDirection.SHORT,
            strategy=StrategyType.PAIRS, confidence=0.75,
            position_size_pct=0.04, entry_price=185.0,
            strategy_source="Pair SHORT: AAPL_MSFT z=+2.3",
            pair_symbol="MSFT",
        ),
        TradeSignal(
            symbol="MSFT", direction=SignalDirection.LONG,
            strategy=StrategyType.PAIRS, confidence=0.75,
            position_size_pct=0.04, entry_price=420.0,
            strategy_source="Pair LONG: AAPL_MSFT z=+2.3",
            pair_symbol="AAPL",
        ),
        TradeSignal(
            symbol="META", direction=SignalDirection.LONG,
            strategy=StrategyType.MEAN_REVERSION, confidence=0.65,
            position_size_pct=0.03, entry_price=550.0,
            stop_price=540.0, target_price=570.0,
            strategy_source="MR LONG: price < BB_low, RSI=25",
        ),
        TradeSignal(
            symbol="XOM", direction=SignalDirection.LONG,
            strategy=StrategyType.MOMENTUM, confidence=0.60,
            position_size_pct=0.025, entry_price=105.0,
            stop_price=100.0, target_price=115.0,
            strategy_source="MOM LONG: pullback to EMA20",
        ),
    ]

    sized = allocator.allocate(test_signals, equity=100_000)

    print(f"\nSized {len(sized)} / {len(test_signals)} signals:\n")
    for sig in sized:
        print(
            f"  {sig.direction.value:>5} {sig.symbol:<6} "
            f"shares={sig.shares:>4} size={sig.position_size_pct:.1%} "
            f"${sig.shares * sig.entry_price:>8,.0f} "
            f"| {sig.strategy_source}"
        )

    alloc = allocator.get_current_allocation()
    print(f"\nCurrent allocation: {alloc}")
    print("\nPortfolio allocator test complete ✅")
