#!/usr/bin/env python3
"""
Comprehensive Backtest for unified_trader.py (AGGRESSIVE MODE)
================================================================

Loads 6 months of historical data for SPY and simulates the full
unified_trader pipeline with aggressive parameter tuning:
  - Lowered composite threshold (0.40 vs 0.55)
  - Relaxed ML hard filter (0.25 vs 0.40)
  - Full-Kelly position sizing (was half-Kelly)
  - Higher Thompson exploration (alpha=2.0)
  - Tighter profit targets for faster rotation
  - Per-arm Thompson attribution (not uniform)

Usage:
    python backtest_unified.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import unified_trader components directly (reuse real production code)
# ---------------------------------------------------------------------------
from unified_trader import (
    UnifiedConfig,
    TechnicalScore,
    CompositeSignal,
    AlpacaRegimeResult,
    ThompsonSampler,
    InlineCircuitBreaker,
    TrackedPosition,
    compute_rsi,
    compute_sma,
    compute_ema,
    compute_macd,
    compute_atr,
    compute_momentum,
    compute_bollinger_position,
    compute_adx,
    bars_to_arrays,
    score_technicals,
    compute_composite_signal,
    _compute_position_size,
    SECTOR_MAP,
    SECTOR_MAX_PCT,
    SECTOR_MAX_POSITIONS,
    get_sector,
)


# ============================================================================
# BACKTEST CONFIG
# ============================================================================

BACKTEST_SYMBOLS = ["SPY"]
INITIAL_CAPITAL = 100_000.0
LOOKBACK_MONTHS = 6
COMMISSION_PER_SHARE = 0.005  # $0.005/share
SLIPPAGE_PCT = 0.0005         # 5 bps slippage


# ============================================================================
# DATA LOADING
# ============================================================================

def download_data(symbols: List[str], months: int = 6) -> Dict[str, pd.DataFrame]:
    """Download historical OHLCV data from yfinance."""
    end = datetime.now()
    # Extra lookback for technical indicator warm-up
    start = end - timedelta(days=months * 30 + 120)

    data = {}
    for sym in symbols:
        print(f"  Downloading {sym}...", end=" ", flush=True)
        try:
            df = yf.download(sym, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            if len(df) >= 60:
                data[sym] = df
                print(f"{len(df)} bars")
            else:
                print(f"SKIPPED (only {len(df)} bars)")
        except Exception as e:
            print(f"FAILED: {e}")
    return data


def df_to_bars(df: pd.DataFrame) -> List[dict]:
    """Convert a pandas DataFrame to the Alpaca-like bars format."""
    bars = []
    for _, row in df.iterrows():
        bars.append({
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row["volume"]),
        })
    return bars


# ============================================================================
# SIMULATED REGIME DETECTION (from price data, no API call)
# ============================================================================

def detect_regime_from_bars(bars: List[dict]) -> AlpacaRegimeResult:
    """Detect regime from bar data — mirrors unified_trader fallback logic."""
    if not bars or len(bars) < 50:
        return AlpacaRegimeResult("neutral", 0.4, {"source": "backtest_fallback"})

    closes = np.array([float(b["c"]) for b in bars])
    highs = np.array([float(b["h"]) for b in bars])
    lows = np.array([float(b["l"]) for b in bars])
    price = closes[-1]

    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, min(200, len(closes)))
    atr = compute_atr(highs, lows, closes)
    atr_pct = atr / price if price > 0 else 0.02

    if atr_pct > 0.03:
        regime, conf = "high_volatility", 0.6
    elif price > sma200 and sma50 > sma200 and price > sma50:
        regime, conf = "trending_bull", 0.7
    elif price > sma200:
        regime, conf = "mean_reverting", 0.5
    elif price < sma200 and sma50 < sma200:
        regime, conf = "trending_bear", 0.65
    else:
        regime, conf = "neutral", 0.4

    return AlpacaRegimeResult(regime, conf, {"source": "backtest_sma"})


# ============================================================================
# SIMULATED ML CONFIDENCE (synthetic — no live model)
# ============================================================================

def simulate_ml_confidence(closes: np.ndarray) -> float:
    """
    Simulate ML confidence from price data (aggressive — higher base, wider range).
    Uses momentum signals + RSI for directional bias.
    """
    if len(closes) < 20:
        return 0.55  # higher default

    rsi = compute_rsi(closes)
    mom_5 = (closes[-1] / closes[-6] - 1) if len(closes) > 6 else 0.0
    mom_20 = (closes[-1] / closes[-21] - 1) if len(closes) > 21 else 0.0
    mom_10 = (closes[-1] / closes[-11] - 1) if len(closes) > 11 else 0.0

    # More aggressive base — bias towards confidence
    base = 0.55 + mom_5 * 3.0 + mom_10 * 1.5 + mom_20 * 0.5
    # RSI: oversold = strong buy signal, overbought = mild reduce
    if rsi < 30:
        base += 0.25
    elif rsi < 40:
        base += 0.15
    elif rsi > 70:
        base -= 0.10
    elif rsi > 60:
        base -= 0.05

    # Less noise for more consistent signals
    noise = np.random.normal(0, 0.06)
    conf = float(np.clip(base + noise, 0.05, 0.95))
    return conf


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@dataclass
class BacktestResult:
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades_blocked_by_ml: int
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_consecutive_losses: int
    avg_holding_days: float
    final_equity: float
    equity_curve: np.ndarray
    trade_log: List[dict]
    thompson_stats: dict
    regime_counts: dict
    daily_returns: np.ndarray


@dataclass
class BacktestPosition:
    symbol: str
    entry_price: float
    entry_idx: int     # bar index
    qty: int
    stop_price: float
    target_price: float
    trailing_stop: float = 0.0
    trailing_active: bool = False
    highest_price: float = 0.0
    atr_at_entry: float = 0.0
    composite_score: float = 0.0
    ml_confidence: float = 0.0
    position_cost: float = 0.0

    def update_trailing(self, current_price: float, trail_pct: float = 0.015,
                        activation_pct: float = 0.03):
        self.highest_price = max(self.highest_price, current_price)
        gain_pct = (current_price - self.entry_price) / self.entry_price
        if gain_pct >= activation_pct:
            self.trailing_active = True
            new_trail = current_price * (1 - trail_pct)
            self.trailing_stop = max(self.trailing_stop, new_trail)

    @property
    def effective_stop(self) -> float:
        if self.trailing_active:
            return max(self.stop_price, self.trailing_stop)
        return self.stop_price


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    cfg: UnifiedConfig = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> BacktestResult:
    """
    Run full backtest simulating unified_trader logic day-by-day.
    """
    cfg = cfg or UnifiedConfig()
    np.random.seed(42)

    # ── Determine common date range ──────────────────────────────
    # Use the intersection of all symbols' dates
    date_sets = [set(df.index) for df in all_data.values()]
    common_dates = sorted(set.intersection(*date_sets))
    if len(common_dates) < 100:
        raise ValueError(f"Only {len(common_dates)} common dates — need at least 100")

    # Use last N months for actual trading, first part for warm-up
    trade_start_idx = 100  # 100-bar warm-up
    trade_dates = common_dates[trade_start_idx:]
    print(f"  Trading period: {trade_dates[0].strftime('%Y-%m-%d')} → "
          f"{trade_dates[-1].strftime('%Y-%m-%d')} ({len(trade_dates)} days)")

    # ── State ────────────────────────────────────────────────────
    equity = initial_capital
    cash = initial_capital
    positions: Dict[str, BacktestPosition] = {}
    trade_log: List[dict] = []
    equity_curve = [initial_capital]
    daily_equity = []

    # Counters
    trades_blocked_ml = 0
    win_count = 0
    loss_count = 0
    total_win_return = 0.0
    total_loss_return = 0.0
    consecutive_losses = 0
    max_consecutive_losses = 0
    holding_days_total = 0
    regime_counts: Dict[str, int] = {}

    # Thompson Sampling
    thompson = ThompsonSampler(
        arms=["technical", "regime", "ml", "tda"],
        prior_alpha=cfg.thompson_prior_alpha,
        prior_beta=cfg.thompson_prior_beta,
    )

    # Circuit breaker
    breaker = InlineCircuitBreaker(max_daily_loss_pct=cfg.max_daily_loss_pct)
    breaker_triggered_days = 0

    # ── Day-by-day simulation ────────────────────────────────────
    for day_idx, trade_date in enumerate(trade_dates):
        # Build bar windows up to this date
        bar_windows: Dict[str, List[dict]] = {}
        for sym in all_data:
            df = all_data[sym]
            mask = df.index <= trade_date
            window_df = df.loc[mask].tail(cfg.bars_lookback)
            if len(window_df) >= 50:
                bar_windows[sym] = df_to_bars(window_df)

        if not bar_windows:
            equity_curve.append(equity)
            continue

        # Get current prices
        current_prices = {}
        for sym, bars in bar_windows.items():
            current_prices[sym] = float(bars[-1]["c"])

        # ── Update portfolio equity ──────────────────────────────
        position_value = sum(
            pos.qty * current_prices.get(sym, pos.entry_price)
            for sym, pos in positions.items()
        )
        equity = cash + position_value
        equity_curve.append(equity)

        # ── Circuit breaker ──────────────────────────────────────
        allowed, reason = breaker.check(equity)
        if not allowed:
            breaker_triggered_days += 1
            # Still manage stops even if halted
            _manage_stops(positions, current_prices, cfg, trade_log,
                          lambda pnl_pct, sym: None, day_idx, trade_dates)
            continue

        # New day reset
        if day_idx == 0 or trade_dates[day_idx].day != trade_dates[day_idx - 1].day:
            breaker.reset_daily(equity)

        # ── Regime detection (use SPY bars) ──────────────────────
        spy_bars = bar_windows.get("SPY", list(bar_windows.values())[0])
        regime = detect_regime_from_bars(spy_bars)
        regime_counts[regime.regime] = regime_counts.get(regime.regime, 0) + 1

        # ── Skip new entries if bearish (but allow in neutral) ──────────
        # Aggressive: only skip in strong bear / crisis
        skip_new_longs = regime.regime in ("crisis", "strong_bear")

        # ── Manage existing positions ────────────────────────────
        to_close = []
        for sym, pos in list(positions.items()):
            price = current_prices.get(sym)
            if price is None:
                continue

            pos.update_trailing(price, cfg.trailing_stop_pct,
                                cfg.trailing_stop_activation)
            effective_stop = pos.effective_stop
            gain_pct = (price - pos.entry_price) / pos.entry_price

            # Stop loss
            if price <= effective_stop:
                reason_str = "trailing stop" if pos.trailing_active else "ATR stop"
                to_close.append((sym, price, reason_str, gain_pct))
                continue

            # Profit target
            if price >= pos.target_price:
                to_close.append((sym, price, "profit target", gain_pct))
                continue

        # Execute closes
        for sym, price, reason_str, gain_pct in to_close:
            pos = positions[sym]
            # Apply slippage + commission
            exit_price = price * (1 - SLIPPAGE_PCT)
            commission = COMMISSION_PER_SHARE * pos.qty * 2  # entry + exit
            pnl_gross = pos.qty * (exit_price - pos.entry_price)
            pnl_net = pnl_gross - commission
            pnl_pct = pnl_net / pos.position_cost if pos.position_cost > 0 else 0

            cash += pos.position_cost + pnl_net  # Return capital + P&L
            holding_days = day_idx - pos.entry_idx
            holding_days_total += max(holding_days, 1)

            if pnl_pct > 0:
                win_count += 1
                total_win_return += pnl_pct
                consecutive_losses = 0
            else:
                loss_count += 1
                total_loss_return += abs(pnl_pct)
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

            # Update Thompson — per-arm attribution based on which arm contributed most
            won = pnl_pct > 0
            # Identify which signal source contributed most to this trade
            tech_contrib = pos.composite_score  # proxy for tech
            ml_contrib = pos.ml_confidence
            dominant_arm = "technical" if tech_contrib > ml_contrib else "ml"
            # Update dominant arm with actual outcome, others with mild update
            thompson.update(dominant_arm, won)
            # Secondary arms get partial update
            for arm in thompson.arms:
                if arm != dominant_arm:
                    # 50% chance to update secondary arms (exploration)
                    if np.random.random() < 0.5:
                        thompson.update(arm, won)

            trade_log.append({
                "symbol": sym,
                "side": "sell",
                "entry_price": pos.entry_price,
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 4),
                "pnl_net": round(pnl_net, 2),
                "reason": reason_str,
                "holding_days": holding_days,
                "composite_score": pos.composite_score,
                "ml_confidence": pos.ml_confidence,
                "day": str(trade_date)[:10],
            })

            del positions[sym]

        # ── Skip new entries if bearish ──────────────────────────
        if skip_new_longs:
            continue
        if len(positions) >= cfg.max_open_positions:
            continue

        # ── Thompson sampling weights ────────────────────────────
        thompson_weights = thompson.sample_weights()

        # ── Score all candidates ─────────────────────────────────
        sector_exposure: Dict[str, float] = {}
        sector_counts: Dict[str, int] = {}
        for sym, pos in positions.items():
            sect = get_sector(sym)
            sector_exposure[sect] = sector_exposure.get(sect, 0) + pos.position_cost
            sector_counts[sect] = sector_counts.get(sect, 0) + 1

        signals: List[CompositeSignal] = []
        for sym in BACKTEST_SYMBOLS:
            if sym in positions:
                continue
            bars = bar_windows.get(sym)
            if not bars or len(bars) < 50:
                continue

            tech = score_technicals(sym, bars, cfg)
            if tech is None:
                continue

            # Simulated ML confidence
            closes = np.array([float(b["c"]) for b in bars])
            ml_conf = simulate_ml_confidence(closes)

            # TDA score: simulate as slight positive (topology neutral)
            tda_score = float(np.random.normal(0.1, 0.15))
            tda_score = float(np.clip(tda_score, -1.0, 1.0))

            # Build current position values dict for sector check
            current_pos_values = {
                s: p.position_cost for s, p in positions.items()
            }

            sig = compute_composite_signal(
                sym, tech, regime, tda_score, ml_conf,
                cfg, equity, current_pos_values,
                thompson_weights=thompson_weights,
            )

            # ── Track ML Hard Filter blocks ────────────────────────
            # The filter is applied inside compute_composite_signal:
            # if ml_conf < ml_min_confidence, BUY becomes HOLD
            if cfg.ml_hard_filter and sig.direction == "HOLD" and ml_conf < cfg.ml_min_confidence:
                # This was likely a BUY that got blocked
                if tech.score > 0.5 and regime.is_bullish:
                    trades_blocked_ml += 1

            if sig.direction == "BUY" and sig.composite_score >= cfg.min_composite_score:
                # Sector cap check
                sect = get_sector(sym)
                sect_exp = sector_exposure.get(sect, 0)
                sect_cnt = sector_counts.get(sect, 0)
                max_sect_dollars = equity * SECTOR_MAX_PCT

                if sect_cnt >= SECTOR_MAX_POSITIONS:
                    continue
                proposed_cost = equity * sig.position_size_pct
                if sect_exp + proposed_cost > max_sect_dollars:
                    continue

                signals.append(sig)

        # Sort by composite score
        signals.sort(key=lambda s: s.composite_score, reverse=True)

        # ── Execute top signals ──────────────────────────────────
        for sig in signals:
            if len(positions) >= cfg.max_open_positions:
                break

            proposed_cost = equity * sig.position_size_pct
            if proposed_cost > cash * 0.95:
                continue
            if proposed_cost < 500:  # Min trade size
                continue

            # Apply slippage on entry
            entry_price = sig.price * (1 + SLIPPAGE_PCT)
            qty = int(proposed_cost / entry_price)
            if qty <= 0:
                continue
            actual_cost = qty * entry_price

            positions[sig.symbol] = BacktestPosition(
                symbol=sig.symbol,
                entry_price=entry_price,
                entry_idx=day_idx,
                qty=qty,
                stop_price=sig.stop_price,
                target_price=round(sig.price * (1 + cfg.profit_target_pct), 2),
                highest_price=sig.price,
                atr_at_entry=sig.atr,
                composite_score=sig.composite_score,
                ml_confidence=sig.ml_confidence,
                position_cost=actual_cost,
            )
            cash -= actual_cost

            # Update sector tracking
            sect = get_sector(sig.symbol)
            sector_exposure[sect] = sector_exposure.get(sect, 0) + actual_cost
            sector_counts[sect] = sector_counts.get(sect, 0) + 1

            trade_log.append({
                "symbol": sig.symbol,
                "side": "buy",
                "entry_price": round(entry_price, 2),
                "composite_score": round(sig.composite_score, 3),
                "ml_confidence": round(sig.ml_confidence, 3),
                "position_size_pct": round(sig.position_size_pct, 4),
                "stop_price": sig.stop_price,
                "target_price": positions[sig.symbol].target_price,
                "thompson_weights": {k: round(v, 3) for k, v in thompson_weights.items()},
                "day": str(trade_date)[:10],
            })

    # ── Final mark-to-market ─────────────────────────────────────
    # Close any remaining positions at last price
    for sym, pos in list(positions.items()):
        last_bars = bar_windows.get(sym)
        if last_bars:
            last_price = float(last_bars[-1]["c"])
        else:
            last_price = pos.entry_price
        pnl = pos.qty * (last_price - pos.entry_price)
        cash += pos.position_cost + pnl
        pnl_pct = pnl / pos.position_cost if pos.position_cost > 0 else 0
        if pnl_pct > 0:
            win_count += 1
            total_win_return += pnl_pct
        else:
            loss_count += 1
            total_loss_return += abs(pnl_pct)

    equity = cash
    equity_curve.append(equity)

    # ── Compute metrics ──────────────────────────────────────────
    eq = np.array(equity_curve)
    daily_returns = np.diff(eq) / (eq[:-1] + 1e-10)

    total_return = (eq[-1] - initial_capital) / initial_capital
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(252)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / (running_max + 1e-10)
    max_dd = float(np.min(drawdowns))

    total_trades = win_count + loss_count
    win_rate = win_count / max(total_trades, 1)
    avg_win = total_win_return / max(win_count, 1)
    avg_loss = total_loss_return / max(loss_count, 1)
    profit_factor = total_win_return / max(total_loss_return, 1e-10)
    avg_holding = holding_days_total / max(total_trades, 1)

    return BacktestResult(
        total_return_pct=round(total_return * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        win_rate=round(win_rate, 4),
        total_trades=total_trades,
        winning_trades=win_count,
        losing_trades=loss_count,
        trades_blocked_by_ml=trades_blocked_ml,
        avg_win_pct=round(avg_win * 100, 2),
        avg_loss_pct=round(avg_loss * 100, 2),
        profit_factor=round(profit_factor, 3),
        max_consecutive_losses=max_consecutive_losses,
        avg_holding_days=round(avg_holding, 1),
        final_equity=round(equity, 2),
        equity_curve=eq,
        trade_log=trade_log,
        thompson_stats=thompson.stats(),
        regime_counts=regime_counts,
        daily_returns=daily_returns,
    )


def _manage_stops(positions, current_prices, cfg, trade_log, on_close_cb,
                  day_idx, trade_dates):
    """Manage stops during circuit breaker halt (close only)."""
    to_close = []
    for sym, pos in list(positions.items()):
        price = current_prices.get(sym)
        if price is None:
            continue
        pos.update_trailing(price, cfg.trailing_stop_pct, cfg.trailing_stop_activation)
        if price <= pos.effective_stop:
            to_close.append(sym)

    for sym in to_close:
        pos = positions[sym]
        price = current_prices.get(sym, pos.entry_price)
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        on_close_cb(pnl_pct, sym)
        del positions[sym]


# ============================================================================
# REPORTING
# ============================================================================

def print_report(result: BacktestResult, buy_hold_returns: Dict[str, float]):
    """Print a comprehensive backtest report."""
    W = 62

    print(f"\n{'=' * W}")
    print(f"{'UNIFIED TRADER BACKTEST RESULTS':^{W}}")
    print(f"{'=' * W}")

    # Performance
    print(f"\n{'── Performance ──':─<{W}}")
    print(f"  {'Initial Capital:':<28} ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  {'Final Equity:':<28} ${result.final_equity:>12,.2f}")
    print(f"  {'Total Return:':<28} {result.total_return_pct:>12.2f}%")
    print(f"  {'Sharpe Ratio:':<28} {result.sharpe_ratio:>12.3f}")
    print(f"  {'Max Drawdown:':<28} {result.max_drawdown_pct:>12.2f}%")
    print(f"  {'Profit Factor:':<28} {result.profit_factor:>12.3f}")

    # Trade statistics
    print(f"\n{'── Trade Statistics ──':─<{W}}")
    print(f"  {'Total Trades:':<28} {result.total_trades:>12}")
    print(f"  {'Winning Trades:':<28} {result.winning_trades:>12}")
    print(f"  {'Losing Trades:':<28} {result.losing_trades:>12}")
    print(f"  {'Win Rate:':<28} {result.win_rate * 100:>11.1f}%")
    print(f"  {'Avg Win:':<28} {result.avg_win_pct:>11.2f}%")
    print(f"  {'Avg Loss:':<28} {result.avg_loss_pct:>11.2f}%")
    print(f"  {'Max Consecutive Losses:':<28} {result.max_consecutive_losses:>12}")
    print(f"  {'Avg Holding Period:':<28} {result.avg_holding_days:>10.1f} d")

    # ML filter
    print(f"\n{'── ML Hard Filter ──':─<{W}}")
    print(f"  {'Trades Blocked by ML:':<28} {result.trades_blocked_by_ml:>12}")
    total_candidates = result.total_trades + result.trades_blocked_by_ml
    if total_candidates > 0:
        block_rate = result.trades_blocked_by_ml / total_candidates * 100
    else:
        block_rate = 0
    print(f"  {'Block Rate:':<28} {block_rate:>11.1f}%")

    # Thompson Sampling
    print(f"\n{'── Thompson Sampling (posterior) ──':─<{W}}")
    for arm, stats in result.thompson_stats.items():
        bar = "█" * int(stats["mean"] * 30)
        print(f"  {arm:<12} α={stats['alpha']:>5.1f}  β={stats['beta']:>5.1f}"
              f"  mean={stats['mean']:.3f}  {bar}")

    # Regime distribution
    print(f"\n{'── Regime Distribution ──':─<{W}}")
    total_days = sum(result.regime_counts.values())
    for regime, count in sorted(result.regime_counts.items(),
                                key=lambda x: -x[1]):
        pct = count / total_days * 100
        bar = "█" * int(pct / 2)
        print(f"  {regime:<20} {count:>4} days ({pct:>5.1f}%)  {bar}")

    # Buy & Hold comparison
    print(f"\n{'── Buy & Hold Comparison ──':─<{W}}")
    for sym, bh_ret in buy_hold_returns.items():
        delta = result.total_return_pct - bh_ret
        marker = "✅" if delta > 0 else "❌"
        print(f"  {sym:<6} B&H: {bh_ret:>7.2f}%  |  Strategy: {result.total_return_pct:>7.2f}%"
              f"  |  Alpha: {delta:>+7.2f}% {marker}")

    # Recent trades
    sells = [t for t in result.trade_log if t.get("side") == "sell"]
    if sells:
        print(f"\n{'── Last 10 Closed Trades ──':─<{W}}")
        for t in sells[-10:]:
            emoji = "🟢" if t.get("pnl_pct", 0) > 0 else "🔴"
            print(f"  {emoji} {t['symbol']:<5} {t.get('day',''):<10} "
                  f"entry=${t['entry_price']:.2f} → exit=${t['exit_price']:.2f} "
                  f"P&L={t['pnl_pct']*100:>+6.2f}%  ({t['reason']})")

    print(f"\n{'=' * W}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 62)
    print("UNIFIED TRADER — COMPREHENSIVE BACKTEST")
    print("=" * 62)
    print(f"  Symbols:  {', '.join(BACKTEST_SYMBOLS)}")
    print(f"  Period:   {LOOKBACK_MONTHS} months")
    print(f"  Capital:  ${INITIAL_CAPITAL:,.0f}")
    print(f"  Slippage: {SLIPPAGE_PCT * 10000:.0f} bps")
    print()

    # ── Download data ────────────────────────────────────────────
    print("Downloading historical data...")
    all_data = download_data(BACKTEST_SYMBOLS, months=LOOKBACK_MONTHS)
    if len(all_data) < 2:
        print("ERROR: Insufficient data downloaded. Exiting.")
        sys.exit(1)
    print(f"  Loaded {len(all_data)} symbols\n")

    # ── Compute buy & hold returns ───────────────────────────────
    buy_hold = {}
    for sym, df in all_data.items():
        # Use last ~6 months
        n = min(len(df), LOOKBACK_MONTHS * 21)  # ~21 trading days/month
        bh = (df["close"].iloc[-1] / df["close"].iloc[-n] - 1) * 100
        buy_hold[sym] = round(bh, 2)

    # ── Run backtest ─────────────────────────────────────────────
    print("Running backtest (AGGRESSIVE mode)...")
    cfg = UnifiedConfig(
        ml_hard_filter=True,
        ml_min_confidence=0.25,       # lowered from 0.40
        thompson_enabled=True,
        thompson_prior_alpha=2.0,
        thompson_prior_beta=1.0,
        max_position_pct=0.08,        # raised from 0.05
        min_position_pct=0.01,
        kelly_fraction=1.00,          # full Kelly (was 0.50)
        default_position_pct=0.04,
        max_daily_loss_pct=0.04,      # wider circuit breaker
        profit_target_pct=0.05,       # tighter profit target for faster rotation
        trailing_stop_activation=0.02,
        trailing_stop_pct=0.012,
        min_composite_score=0.40,     # lowered from 0.55
        bars_lookback=100,
    )

    result = run_backtest(all_data, cfg, INITIAL_CAPITAL)

    # ── Print report ─────────────────────────────────────────────
    print_report(result, buy_hold)

    # ── Save results to JSON ─────────────────────────────────────
    import json
    output = {
        "timestamp": datetime.now().isoformat(),
        "symbols": BACKTEST_SYMBOLS,
        "period_months": LOOKBACK_MONTHS,
        "initial_capital": INITIAL_CAPITAL,
        "total_return_pct": result.total_return_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "trades_blocked_by_ml": result.trades_blocked_by_ml,
        "profit_factor": result.profit_factor,
        "avg_win_pct": result.avg_win_pct,
        "avg_loss_pct": result.avg_loss_pct,
        "max_consecutive_losses": result.max_consecutive_losses,
        "avg_holding_days": result.avg_holding_days,
        "final_equity": result.final_equity,
        "thompson_stats": result.thompson_stats,
        "regime_counts": result.regime_counts,
        "config": {
            "ml_hard_filter": cfg.ml_hard_filter,
            "ml_min_confidence": cfg.ml_min_confidence,
            "thompson_enabled": cfg.thompson_enabled,
            "max_position_pct": cfg.max_position_pct,
            "kelly_fraction": cfg.kelly_fraction,
            "profit_target_pct": cfg.profit_target_pct,
            "trailing_stop_pct": cfg.trailing_stop_pct,
            "min_composite_score": cfg.min_composite_score,
        },
    }
    with open("results/backtest_unified_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/backtest_unified_results.json")


if __name__ == "__main__":
    main()
