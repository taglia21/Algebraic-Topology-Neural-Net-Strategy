#!/usr/bin/env python3
"""
6-Month HMM + GARCH Backtest
=============================================================

Loads historical OHLCV data for the strategy universe, runs
strategy_engine with HMM regime detection and GARCH volatility,
and produces institutional-grade performance metrics.

Tracks: Sharpe ratio, max drawdown, win rate, profit factor,
total return, and compares vs buy-and-hold SPY benchmark.

Usage:
    python backtest_hmm_garch.py
"""

import sys
import os
import json
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from strategy_engine import (
    StrategyEngine,
    EngineConfig,
    TradeSignal,
    SignalDirection,
    StrategyType,
    compute_atr,
)

logger = logging.getLogger("backtest")

# ============================================================================
# CONFIG
# ============================================================================

# Diversified universe matching strategy_engine's sector coverage
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    # Financials
    "JPM", "GS", "BAC",
    # Energy
    "XOM", "CVX",
    # Healthcare
    "UNH", "JNJ",
    # Consumer
    "AMZN", "HD",
    # Industrials
    "CAT", "HON",
    # Benchmark
    "SPY",
]

INITIAL_CAPITAL = 100_000.0
LOOKBACK_MONTHS = 6
WARMUP_BARS = 260          # Need 260 bars for 200-SMA + HMM warmup
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5           # 5 bps per side
MAX_POSITIONS = 10


# ============================================================================
# DATA
# ============================================================================

def download_universe(symbols: List[str], months: int = 6) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data with extra lookback for indicator warmup."""
    end = datetime.now()
    start = end - timedelta(days=months * 30 + WARMUP_BARS + 30)  # Extra padding

    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            if len(df) >= 100:
                data[sym] = df
                print(f"  {sym}: {len(df)} bars")
            else:
                print(f"  {sym}: SKIP ({len(df)} bars)")
        except Exception as e:
            print(f"  {sym}: FAIL ({e})")
    return data


def build_price_volume_frames(
    all_data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Build aligned price, volume, and per-symbol OHLCV frames."""
    price_frames = {}
    volume_frames = {}

    for sym, df in all_data.items():
        price_frames[sym] = df["close"]
        volume_frames[sym] = df["volume"]

    price_data = pd.DataFrame(price_frames).dropna(how="all")
    volume_data = pd.DataFrame(volume_frames).reindex(price_data.index).fillna(0)

    # OHLCV dicts (per symbol) - for ATR/ADX computation inside strategy_engine
    ohlcv_data: Dict[str, pd.DataFrame] = {}
    for sym, df in all_data.items():
        ohlcv_data[sym] = df.reindex(price_data.index).ffill()

    return price_data, volume_data, ohlcv_data


# ============================================================================
# BACKTEST POSITION
# ============================================================================

@dataclass
class Position:
    symbol: str
    direction: str            # "long" or "short"
    entry_price: float
    qty: int
    stop_price: float
    target_price: float
    strategy: str
    entry_day: int
    cost: float
    confidence: float = 0.0
    regime: str = ""
    garch_vol: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop: float = 0.0
    max_hold_days: int = 0

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return self.qty * (current_price - self.entry_price)
        else:
            return self.qty * (self.entry_price - current_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        return self.unrealized_pnl(current_price) / max(self.cost, 1)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@dataclass
class BacktestMetrics:
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_holding_days: float = 0.0
    max_consecutive_losses: int = 0
    final_equity: float = 0.0
    spy_buy_hold_pct: float = 0.0
    alpha_pct: float = 0.0
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_breakdown: Dict[str, dict] = field(default_factory=dict)


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[BacktestMetrics, np.ndarray, List[dict]]:
    """
    Day-by-day backtest using StrategyEngine with HMM+GARCH.

    Returns (metrics, equity_curve, trade_log).
    """
    # Build aligned frames
    price_data, volume_data, ohlcv_data = build_price_volume_frames(all_data)
    all_dates = list(price_data.index)

    if len(all_dates) < WARMUP_BARS + 20:
        raise ValueError(f"Need at least {WARMUP_BARS + 20} bars, got {len(all_dates)}")

    # Determine trading period (after warmup)
    trade_start = WARMUP_BARS
    trade_dates = all_dates[trade_start:]

    # Limit to last ~6 months of actual trading
    target_days = LOOKBACK_MONTHS * 21  # ~21 trading days per month
    if len(trade_dates) > target_days:
        trade_dates = trade_dates[-target_days:]
        trade_start = all_dates.index(trade_dates[0])

    print(f"  Warmup: {all_dates[0].strftime('%Y-%m-%d')} → {all_dates[trade_start-1].strftime('%Y-%m-%d')}")
    print(f"  Trading: {trade_dates[0].strftime('%Y-%m-%d')} → {trade_dates[-1].strftime('%Y-%m-%d')} ({len(trade_dates)} days)")

    # Initialize engine with HMM + GARCH enabled
    cfg = EngineConfig(
        use_hmm_regime=True,
        use_garch_stops=True,
        use_ml_stacker=True,
        use_order_flow=True,
        use_adaptive_params=True,
        min_confidence=0.45,
        max_position_pct=0.06,
    )
    engine = StrategyEngine(cfg)

    # State
    cash = initial_capital
    positions: Dict[str, Position] = {}
    equity_curve = [initial_capital]
    trade_log: List[dict] = []
    regime_counts: Dict[str, int] = {}
    strategy_stats: Dict[str, dict] = {
        "pairs_trading": {"wins": 0, "losses": 0, "gross_pnl": 0.0},
        "mean_reversion": {"wins": 0, "losses": 0, "gross_pnl": 0.0},
        "momentum_regime": {"wins": 0, "losses": 0, "gross_pnl": 0.0},
    }

    total_wins = 0
    total_losses = 0
    total_win_return = 0.0
    total_loss_return = 0.0
    holding_days_sum = 0
    consecutive_losses = 0
    max_consec_losses = 0

    # ── Day-by-day simulation ─────────────────────────────────
    for day_i, trade_date in enumerate(trade_dates):
        date_idx = all_dates.index(trade_date)

        # Price window up to today (for strategy engine)
        window_end = date_idx + 1
        price_window = price_data.iloc[:window_end]
        vol_window = volume_data.iloc[:window_end]

        # Build OHLCV window per symbol
        ohlcv_window: Dict[str, pd.DataFrame] = {}
        for sym in ohlcv_data:
            ohlcv_window[sym] = ohlcv_data[sym].iloc[:window_end]

        # Current prices
        current_prices = {}
        for sym in price_data.columns:
            p = price_data[sym].iloc[date_idx]
            if not np.isnan(p):
                current_prices[sym] = float(p)

        # ── Mark to market ────────────────────────────────────
        pos_value = 0.0
        for sym, pos in positions.items():
            cp = current_prices.get(sym, pos.entry_price)
            pos_value += pos.cost + pos.unrealized_pnl(cp)
        equity = cash + pos_value
        equity_curve.append(equity)

        # ── Manage existing positions (stops, targets, time) ──
        to_close: List[Tuple[str, float, str]] = []
        for sym, pos in list(positions.items()):
            cp = current_prices.get(sym)
            if cp is None:
                continue

            # Update trailing stop
            if pos.direction == "long":
                pos.highest_price = max(pos.highest_price, cp)
                # Trailing activate at +3%
                gain_pct = (cp - pos.entry_price) / pos.entry_price
                if gain_pct > 0.03:
                    new_trail = cp * 0.985  # 1.5% trailing
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)
            else:
                pos.lowest_price = min(pos.lowest_price, cp) if pos.lowest_price > 0 else cp
                gain_pct = (pos.entry_price - cp) / pos.entry_price
                if gain_pct > 0.03:
                    new_trail = cp * 1.015
                    if pos.trailing_stop > 0:
                        pos.trailing_stop = min(pos.trailing_stop, new_trail)
                    else:
                        pos.trailing_stop = new_trail

            # Check stop loss
            if pos.direction == "long":
                eff_stop = max(pos.stop_price, pos.trailing_stop) if pos.trailing_stop > 0 else pos.stop_price
                if cp <= eff_stop:
                    to_close.append((sym, cp, "stop_loss"))
                    continue
            else:
                eff_stop = min(pos.stop_price, pos.trailing_stop) if pos.trailing_stop > 0 else pos.stop_price
                if eff_stop > 0 and cp >= eff_stop:
                    to_close.append((sym, cp, "stop_loss"))
                    continue

            # Check target
            if pos.direction == "long" and pos.target_price > 0 and cp >= pos.target_price:
                to_close.append((sym, cp, "target"))
                continue
            if pos.direction == "short" and pos.target_price > 0 and cp <= pos.target_price:
                to_close.append((sym, cp, "target"))
                continue

            # Time exit
            if pos.max_hold_days > 0 and day_i - pos.entry_day >= pos.max_hold_days:
                to_close.append((sym, cp, "time_exit"))
                continue

        # Execute closes
        for sym, exit_price, reason in to_close:
            pos = positions[sym]
            slippage = exit_price * SLIPPAGE_BPS / 10000
            if pos.direction == "long":
                net_exit = exit_price - slippage
            else:
                net_exit = exit_price + slippage
            commission = COMMISSION_PER_SHARE * pos.qty * 2

            pnl = pos.unrealized_pnl(net_exit) - commission
            pnl_pct = pnl / max(pos.cost, 1)

            cash += pos.cost + pnl
            holding_days = day_i - pos.entry_day
            holding_days_sum += max(holding_days, 1)

            strat = pos.strategy
            if strat in strategy_stats:
                strategy_stats[strat]["gross_pnl"] += pnl

            if pnl > 0:
                total_wins += 1
                total_win_return += pnl_pct
                consecutive_losses = 0
                if strat in strategy_stats:
                    strategy_stats[strat]["wins"] += 1
            else:
                total_losses += 1
                total_loss_return += abs(pnl_pct)
                consecutive_losses += 1
                max_consec_losses = max(max_consec_losses, consecutive_losses)
                if strat in strategy_stats:
                    strategy_stats[strat]["losses"] += 1

            # Feed back to engine for adaptive tuning
            engine.record_trade_result(
                strategy=strat,
                pnl=pnl,
                symbol=sym,
                pnl_pct=pnl_pct,
                holding_bars=holding_days,
                regime=pos.regime,
                exit_reason=reason,
            )

            trade_log.append({
                "symbol": sym,
                "direction": pos.direction,
                "strategy": strat,
                "entry_price": round(pos.entry_price, 2),
                "exit_price": round(net_exit, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "reason": reason,
                "holding_days": holding_days,
                "regime": pos.regime,
                "garch_vol": round(pos.garch_vol, 4) if pos.garch_vol else 0,
                "confidence": round(pos.confidence, 3),
                "date": str(trade_date)[:10],
            })

            del positions[sym]

        # ── Generate new signals ───────────────────────────────
        if len(positions) >= MAX_POSITIONS:
            continue

        try:
            signals = engine.get_signals(
                price_data=price_window,
                volume_data=vol_window,
                equity=equity,
                current_positions={s: {"qty": p.qty, "entry_price": p.entry_price}
                                   for s, p in positions.items()},
                ohlcv_data=ohlcv_window,
            )
        except Exception as e:
            logger.debug(f"Signal generation failed on {trade_date}: {e}")
            signals = []

        # Track regime
        regime = engine._current_regime
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Execute top signals
        for sig in signals:
            if len(positions) >= MAX_POSITIONS:
                break
            if sig.symbol in positions:
                continue
            if sig.direction == SignalDirection.CLOSE:
                continue

            cp = current_prices.get(sig.symbol)
            if cp is None or cp <= 0:
                continue

            # Position sizing
            size_pct = min(sig.position_size_pct, cfg.max_position_pct)
            alloc = equity * size_pct
            if alloc > cash * 0.95 or alloc < 500:
                continue

            # Apply slippage on entry
            slippage = cp * SLIPPAGE_BPS / 10000
            if sig.direction == SignalDirection.LONG:
                entry_price = cp + slippage
            else:
                entry_price = cp - slippage

            qty = max(1, int(alloc / entry_price))
            actual_cost = qty * entry_price

            if actual_cost > cash * 0.95:
                continue

            positions[sig.symbol] = Position(
                symbol=sig.symbol,
                direction=sig.direction.value,
                entry_price=entry_price,
                qty=qty,
                stop_price=sig.stop_price,
                target_price=sig.target_price,
                strategy=sig.strategy.value,
                entry_day=day_i,
                cost=actual_cost,
                confidence=sig.confidence,
                regime=sig.regime,
                garch_vol=sig.garch_vol if sig.garch_vol else 0.0,
                highest_price=entry_price,
                lowest_price=entry_price,
                max_hold_days=sig.max_hold_days,
            )
            cash -= actual_cost

    # ── Final: close remaining positions at last price ────────
    for sym, pos in list(positions.items()):
        cp = current_prices.get(sym, pos.entry_price)
        pnl = pos.unrealized_pnl(cp)
        cash += pos.cost + pnl
        pnl_pct = pnl / max(pos.cost, 1)
        if pnl > 0:
            total_wins += 1
            total_win_return += pnl_pct
        else:
            total_losses += 1
            total_loss_return += abs(pnl_pct)

    final_equity = cash
    equity_curve.append(final_equity)
    eq = np.array(equity_curve)

    # ── Compute metrics ────────────────────────────────────────
    daily_returns = np.diff(eq) / (eq[:-1] + 1e-10)

    total_return = (final_equity - initial_capital) / initial_capital
    trading_days = len(trade_dates)
    ann_factor = 252 / max(trading_days, 1)
    ann_return = (1 + total_return) ** ann_factor - 1

    mean_dr = float(np.mean(daily_returns))
    std_dr = float(np.std(daily_returns)) + 1e-10
    sharpe = (mean_dr / std_dr) * np.sqrt(252)

    downside = daily_returns[daily_returns < 0]
    downside_std = float(np.std(downside)) + 1e-10 if len(downside) > 0 else 1e-10
    sortino = (mean_dr / downside_std) * np.sqrt(252)

    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / (running_max + 1e-10)
    max_dd = float(np.min(drawdowns))

    total_trades = total_wins + total_losses
    win_rate = total_wins / max(total_trades, 1)
    avg_win = total_win_return / max(total_wins, 1)
    avg_loss = total_loss_return / max(total_losses, 1)
    profit_factor = total_win_return / max(total_loss_return, 1e-10)
    avg_hold = holding_days_sum / max(total_trades, 1)

    # SPY buy & hold
    spy_bh = 0.0
    if "SPY" in all_data:
        spy_df = all_data["SPY"]
        spy_prices = spy_df["close"].dropna()
        if len(spy_prices) > target_days:
            spy_start_price = float(spy_prices.iloc[-target_days])
        else:
            spy_start_price = float(spy_prices.iloc[0])
        spy_end_price = float(spy_prices.iloc[-1])
        spy_bh = (spy_end_price / spy_start_price - 1) * 100

    # Strategy breakdown
    strat_breakdown = {}
    for strat, stats in strategy_stats.items():
        s_total = stats["wins"] + stats["losses"]
        strat_breakdown[strat] = {
            "trades": s_total,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats["wins"] / max(s_total, 1),
            "gross_pnl": round(stats["gross_pnl"], 2),
        }

    metrics = BacktestMetrics(
        total_return_pct=round(total_return * 100, 2),
        annualized_return_pct=round(ann_return * 100, 2),
        sharpe_ratio=round(float(sharpe), 3),
        sortino_ratio=round(float(sortino), 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3),
        total_trades=total_trades,
        winning_trades=total_wins,
        losing_trades=total_losses,
        avg_win_pct=round(avg_win * 100, 2),
        avg_loss_pct=round(avg_loss * 100, 2),
        avg_holding_days=round(avg_hold, 1),
        max_consecutive_losses=max_consec_losses,
        final_equity=round(final_equity, 2),
        spy_buy_hold_pct=round(spy_bh, 2),
        alpha_pct=round(total_return * 100 - spy_bh, 2),
        regime_distribution=regime_counts,
        strategy_breakdown=strat_breakdown,
    )

    return metrics, eq, trade_log


# ============================================================================
# REPORTING
# ============================================================================

def print_report(m: BacktestMetrics, trade_log: List[dict]):
    W = 64
    print(f"\n{'=' * W}")
    print(f"{'HMM + GARCH STRATEGY BACKTEST':^{W}}")
    print(f"{'=' * W}")

    print(f"\n── Performance {'─' * (W - 15)}")
    print(f"  {'Initial Capital:':<30} ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  {'Final Equity:':<30} ${m.final_equity:>12,.2f}")
    print(f"  {'Total Return:':<30} {m.total_return_pct:>12.2f}%")
    print(f"  {'Annualized Return:':<30} {m.annualized_return_pct:>12.2f}%")
    print(f"  {'Sharpe Ratio:':<30} {m.sharpe_ratio:>12.3f}")
    print(f"  {'Sortino Ratio:':<30} {m.sortino_ratio:>12.3f}")
    print(f"  {'Max Drawdown:':<30} {m.max_drawdown_pct:>12.2f}%")
    print(f"  {'Profit Factor:':<30} {m.profit_factor:>12.3f}")

    print(f"\n── Trade Statistics {'─' * (W - 20)}")
    print(f"  {'Total Trades:':<30} {m.total_trades:>12}")
    print(f"  {'Winners:':<30} {m.winning_trades:>12}")
    print(f"  {'Losers:':<30} {m.losing_trades:>12}")
    print(f"  {'Win Rate:':<30} {m.win_rate * 100:>11.1f}%")
    print(f"  {'Avg Win:':<30} {m.avg_win_pct:>11.2f}%")
    print(f"  {'Avg Loss:':<30} {m.avg_loss_pct:>11.2f}%")
    print(f"  {'Avg Holding (days):':<30} {m.avg_holding_days:>12.1f}")
    print(f"  {'Max Consecutive Losses:':<30} {m.max_consecutive_losses:>12}")

    print(f"\n── Strategy Breakdown {'─' * (W - 22)}")
    for strat, stats in m.strategy_breakdown.items():
        wr = stats['win_rate'] * 100
        print(f"  {strat:<22} trades={stats['trades']:>3}  "
              f"WR={wr:>5.1f}%  P&L=${stats['gross_pnl']:>+10,.2f}")

    print(f"\n── Regime Distribution {'─' * (W - 23)}")
    total_days = sum(m.regime_distribution.values()) or 1
    for reg, cnt in sorted(m.regime_distribution.items(), key=lambda x: -x[1]):
        pct = cnt / total_days * 100
        bar = "#" * int(pct / 2)
        print(f"  {reg:<22} {cnt:>4} days ({pct:>5.1f}%)  {bar}")

    print(f"\n── Benchmark Comparison {'─' * (W - 24)}")
    print(f"  {'SPY Buy & Hold:':<30} {m.spy_buy_hold_pct:>12.2f}%")
    print(f"  {'Strategy Return:':<30} {m.total_return_pct:>12.2f}%")
    marker = "+" if m.alpha_pct > 0 else ""
    print(f"  {'Alpha:':<30} {marker}{m.alpha_pct:>11.2f}%")

    # Recent trades
    sells = [t for t in trade_log if "pnl" in t]
    if sells:
        print(f"\n── Last 15 Trades {'─' * (W - 18)}")
        for t in sells[-15:]:
            icon = "W" if t["pnl"] > 0 else "L"
            print(f"  [{icon}] {t['symbol']:<6} {t['direction']:<5} "
                  f"${t['entry_price']:.2f}→${t['exit_price']:.2f} "
                  f"P&L={t['pnl_pct']*100:>+6.2f}% "
                  f"({t['reason']}) {t['strategy'][:12]}")

    print(f"\n{'=' * W}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 64)
    print("HMM + GARCH STRATEGY ENGINE BACKTEST")
    print("=" * 64)
    print(f"  Universe:  {len(UNIVERSE)} symbols")
    print(f"  Period:    {LOOKBACK_MONTHS} months")
    print(f"  Capital:   ${INITIAL_CAPITAL:,.0f}")
    print(f"  Slippage:  {SLIPPAGE_BPS} bps per side")
    print()

    # Download data
    print("Downloading historical data...")
    all_data = download_universe(UNIVERSE, months=LOOKBACK_MONTHS)
    if len(all_data) < 5:
        print("ERROR: Insufficient data. Need at least 5 symbols.")
        sys.exit(1)
    print(f"\nLoaded {len(all_data)} symbols\n")

    # Run backtest
    print("Running backtest with HMM regime + GARCH volatility...")
    metrics, eq_curve, trade_log = run_backtest(all_data, INITIAL_CAPITAL)

    # Print report
    print_report(metrics, trade_log)

    # Save to JSON
    Path("results").mkdir(exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "universe": UNIVERSE,
            "period_months": LOOKBACK_MONTHS,
            "initial_capital": INITIAL_CAPITAL,
            "slippage_bps": SLIPPAGE_BPS,
            "hmm_enabled": True,
            "garch_enabled": True,
        },
        "metrics": {
            "total_return_pct": metrics.total_return_pct,
            "annualized_return_pct": metrics.annualized_return_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "avg_win_pct": metrics.avg_win_pct,
            "avg_loss_pct": metrics.avg_loss_pct,
            "avg_holding_days": metrics.avg_holding_days,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "final_equity": metrics.final_equity,
            "alpha_vs_spy_pct": metrics.alpha_pct,
            "spy_buy_hold_pct": metrics.spy_buy_hold_pct,
        },
        "strategy_breakdown": metrics.strategy_breakdown,
        "regime_distribution": metrics.regime_distribution,
        "trade_log": trade_log[-50:],  # Last 50 trades
    }
    with open("results/backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/backtest_results.json")


if __name__ == "__main__":
    main()
