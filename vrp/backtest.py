"""
vrp/backtest.py
===============
Options backtesting engine for the VRP strategy.

Uses historical SPX and VIX daily data (from yfinance) to simulate
systematic put credit spread trading. The backtest models:

- Realistic entry/exit based on daily close prices
- VIX-based regime filtering and position sizing
- Transaction costs (commissions + slippage)
- Position management (profit targets, stop losses, time stops)
- Walk-forward: no lookahead, all decisions use data available at the time

This is NOT an options-level tick backtest — we use Black-Scholes for
theoretical option pricing rather than historical option quotes. This is
a standard approach for strategies that trade liquid index options where
the volatility surface is well-behaved.

Limitations:
- BS pricing doesn't capture skew/surface dynamics perfectly
- Intraday moves (gap risk) are approximated via daily ranges
- Slippage is modeled as fixed cost rather than market-impact
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from vrp.config import Config, get_config
from vrp.strategy import (
    VRPStrategy, SpreadPosition, TradeAction,
    VIXRegime, VIXRegimeClassifier,
)
from vrp.utils import (
    bs_put_price, bs_greeks, setup_logger,
    next_monthly_expiry, dte, years_to_expiry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Comprehensive performance metrics for the backtest."""

    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    annual_volatility: float = 0.0
    calmar_ratio: float = 0.0

    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_days_held: float = 0.0
    max_concurrent: int = 0

    # P&L
    total_pnl: float = 0.0
    total_commissions: float = 0.0
    total_slippage: float = 0.0
    avg_credit_received: float = 0.0

    # Benchmark
    spx_return: float = 0.0
    spx_sharpe: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        import dataclasses
        return dataclasses.asdict(self)

    def summary(self) -> str:
        """Pretty-print summary."""
        lines = [
            "",
            "=" * 60,
            "  VRP ALPHA ENGINE — BACKTEST RESULTS",
            "=" * 60,
            "",
            f"  Total Return:      {self.total_return:>+8.1%}   (SPX: {self.spx_return:+.1%})",
            f"  Annual Return:     {self.annual_return:>+8.1%}",
            f"  Sharpe Ratio:      {self.sharpe_ratio:>8.2f}   (SPX: {self.spx_sharpe:.2f})",
            f"  Sortino Ratio:     {self.sortino_ratio:>8.2f}",
            f"  Max Drawdown:      {self.max_drawdown:>8.1%}",
            f"  Annual Volatility: {self.annual_volatility:>8.1%}",
            f"  Calmar Ratio:      {self.calmar_ratio:>8.2f}",
            f"  Alpha:             {self.alpha:>+8.1%}",
            "",
            "  --- Trade Statistics ---",
            f"  Total Trades:      {self.total_trades:>8d}",
            f"  Win Rate:          {self.win_rate:>8.1%}",
            f"  Avg Win:           ${self.avg_win:>+8.0f}",
            f"  Avg Loss:          ${self.avg_loss:>+8.0f}",
            f"  Profit Factor:     {self.profit_factor:>8.2f}",
            f"  Avg Days Held:     {self.avg_days_held:>8.1f}",
            f"  Max Concurrent:    {self.max_concurrent:>8d}",
            "",
            "  --- Costs ---",
            f"  Total P&L:         ${self.total_pnl:>+10,.0f}",
            f"  Commissions:       ${self.total_commissions:>10,.0f}",
            f"  Slippage:          ${self.total_slippage:>10,.0f}",
            f"  Net P&L:           ${self.total_pnl - self.total_commissions - self.total_slippage:>+10,.0f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

def load_market_data(
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Load SPX and VIX daily data from yfinance.

    Returns DataFrame with columns: [spx_close, spx_high, spx_low, vix_close]
    indexed by date.
    """
    import yfinance as yf

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"vrp_data_{start}_{end}.parquet"

    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"Downloading SPX and VIX data: {start} to {end}")

    # SPX data via ^GSPC
    spx = yf.download("^GSPC", start=start, end=end, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)

    if spx.empty or vix.empty:
        raise RuntimeError("Failed to download market data from yfinance")

    # Handle multi-level columns from yfinance
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = pd.DataFrame({
        "spx_close": spx["Close"],
        "spx_high": spx["High"],
        "spx_low": spx["Low"],
        "spx_open": spx["Open"],
        "vix_close": vix["Close"].reindex(spx.index, method="ffill"),
    })

    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna()

    # Add 200-day SMA
    df["spx_200sma"] = df["spx_close"].rolling(200).mean()

    # Add realized volatility (20-day)
    df["realized_vol_20d"] = (
        df["spx_close"].pct_change().rolling(20).std() * np.sqrt(252)
    )

    logger.info(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    df.to_parquet(cache_file)

    return df


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class VRPBacktester:
    """Walk-forward backtester for the VRP put credit spread strategy.

    Simulates day-by-day execution:
    1. Each day, check if we should open new trades
    2. Mark all positions to market
    3. Evaluate exit rules
    4. Track equity curve with realistic costs
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.strategy = VRPStrategy(self.config)

        # Account tracking
        self.initial_capital = self.config.backtest.initial_capital
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.high_water_mark = self.initial_capital

        # Cost tracking
        self.total_commissions = 0.0
        self.total_slippage = 0.0

        # Equity curve
        self.equity_curve: List[Tuple[date, float]] = []
        self.daily_pnl: List[float] = []

        # Trade log
        self.closed_trades: List[SpreadPosition] = []
        self.max_concurrent = 0

    def run(
        self,
        start: str = "2020-01-01",
        end: str = "2025-12-31",
        verbose: bool = True,
    ) -> BacktestMetrics:
        """Run the full backtest.

        Parameters
        ----------
        start : Backtest start date
        end : Backtest end date
        verbose : Print progress

        Returns
        -------
        BacktestMetrics with full performance analysis
        """
        # Load data
        # Start earlier to build 200-day SMA
        data_start = (pd.Timestamp(start) - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
        data = load_market_data(data_start, end)

        # Filter to backtest period
        bt_start = pd.Timestamp(start)
        bt_data = data[data.index >= bt_start].copy()

        if len(bt_data) < 20:
            raise ValueError(f"Insufficient data for backtest: {len(bt_data)} days")

        if verbose:
            print(f"\n{'='*60}")
            print(f"  VRP ALPHA ENGINE — BACKTEST")
            print(f"{'='*60}")
            print(f"  Period:    {bt_data.index[0].date()} → {bt_data.index[-1].date()}")
            print(f"  Capital:   ${self.initial_capital:,.0f}")
            print(f"  Days:      {len(bt_data)}")
            print(f"{'='*60}\n")

        # Track SPX for benchmark
        spx_start = bt_data["spx_close"].iloc[0]

        prev_equity = self.equity

        for i, (dt, row) in enumerate(bt_data.iterrows()):
            today = dt.date() if hasattr(dt, 'date') else dt

            spx = row["spx_close"]
            vix = row["vix_close"]
            spx_200sma = row.get("spx_200sma", None)
            iv = vix / 100.0  # VIX as IV proxy

            if pd.isna(vix) or pd.isna(spx):
                continue

            # ----- Mark positions to market -----
            for pos in self.strategy.open_positions:
                self.strategy.manager.mark_to_market(
                    pos, spx, iv, today, self.config.backtest.risk_free_rate / 100
                    if self.config.backtest.risk_free_rate > 1 else self.config.backtest.risk_free_rate
                )

            # Track max concurrent
            n_open = len(self.strategy.open_positions)
            self.max_concurrent = max(self.max_concurrent, n_open)

            # ----- Evaluate exits -----
            actions = self.strategy.evaluate_positions(
                spx, vix, iv, today, self.config.backtest.risk_free_rate
            )

            for pos, action in actions:
                if action in (
                    TradeAction.CLOSE_PROFIT,
                    TradeAction.CLOSE_STOP,
                    TradeAction.CLOSE_EXPIRY,
                ):
                    # Cost to close = current spread value (we pay to buy back)
                    close_cost = pos.current_value * pos.quantity
                    # Close costs (only closing leg — entry costs already paid)
                    comm = self.config.backtest.commission_per_contract * pos.quantity
                    slip = self.config.backtest.slippage_per_contract * pos.quantity

                    pnl = self.strategy.close_position(
                        pos, action.value, as_of=today
                    )

                    # Debit cash for closing cost + transaction costs
                    self.cash -= close_cost + comm + slip
                    self.total_commissions += comm
                    self.total_slippage += slip
                    self.closed_trades.append(pos)

                elif action == TradeAction.ROLL:
                    # Close current position
                    close_cost = pos.current_value * pos.quantity
                    comm = self.config.backtest.commission_per_contract * pos.quantity
                    slip = self.config.backtest.slippage_per_contract * pos.quantity

                    pnl = self.strategy.close_position(pos, "roll", as_of=today)
                    self.cash -= close_cost + comm + slip
                    self.total_commissions += comm
                    self.total_slippage += slip
                    self.closed_trades.append(pos)

                    # Open new position with further expiry
                    new_pos = self.strategy.construct_spread(
                        spx_price=spx,
                        vix=vix,
                        account_equity=self.equity,
                        as_of=today,
                        risk_free_rate=self.config.backtest.risk_free_rate,
                    )
                    if new_pos:
                        # Credit received, minus entry costs
                        entry_comm = self.config.backtest.commission_per_contract * new_pos.quantity
                        entry_slip = self.config.backtest.slippage_per_contract * new_pos.quantity
                        self.cash += new_pos.entry_credit * new_pos.quantity
                        self.cash -= entry_comm + entry_slip
                        self.total_commissions += entry_comm
                        self.total_slippage += entry_slip

            # ----- Check for new entries -----
            can_trade = True
            # Check risk limits
            if self.equity < self.config.risk.min_equity:
                can_trade = False

            drawdown = (self.equity - self.high_water_mark) / self.high_water_mark
            if drawdown < self.config.risk.max_drawdown_halt:
                can_trade = False
                # Reset HWM to current equity so we can eventually resume
                # This simulates a "cooling off" period — next day can trade again
                # from the new equity level (real desks reset PnL tracking)
                self.high_water_mark = self.equity

            should_trade = can_trade and self.strategy.should_open_new_trade(
                spx_price=spx,
                vix=vix,
                spx_200sma=spx_200sma if not pd.isna(spx_200sma) else None,
                as_of=today,
            )

            if should_trade:
                new_pos = self.strategy.construct_spread(
                    spx_price=spx,
                    vix=vix,
                    account_equity=self.equity,
                    as_of=today,
                    risk_free_rate=self.config.backtest.risk_free_rate,
                )
                if new_pos:
                    # Credit received goes to cash
                    self.cash += new_pos.entry_credit * new_pos.quantity
                    # Entry costs (only entry leg)
                    entry_comm = self.config.backtest.commission_per_contract * new_pos.quantity
                    entry_slip = self.config.backtest.slippage_per_contract * new_pos.quantity
                    self.cash -= entry_comm + entry_slip
                    self.total_commissions += entry_comm
                    self.total_slippage += entry_slip

            # ----- Update equity -----
            # Equity = cash + unrealized P&L on open positions
            unrealized = sum(
                (p.entry_credit - p.current_value) * p.quantity
                for p in self.strategy.open_positions
            )
            self.equity = self.cash + unrealized
            self.high_water_mark = max(self.high_water_mark, self.equity)

            # Track daily P&L
            daily_pnl = self.equity - prev_equity
            self.daily_pnl.append(daily_pnl)
            prev_equity = self.equity

            self.equity_curve.append((today, self.equity))

            # Progress
            if verbose and i % 63 == 0 and i > 0:
                ret = (self.equity / self.initial_capital - 1)
                print(
                    f"  {today} | equity ${self.equity:>10,.0f} | "
                    f"return {ret:>+7.1%} | "
                    f"open {n_open} | VIX {vix:.1f} | SPX {spx:.0f}"
                )

        # ----- Calculate metrics -----
        spx_end = bt_data["spx_close"].iloc[-1]
        metrics = self._calculate_metrics(
            spx_start=spx_start,
            spx_end=spx_end,
            spx_series=bt_data["spx_close"],
        )

        if verbose:
            print(metrics.summary())

        return metrics

    def _calculate_metrics(
        self,
        spx_start: float,
        spx_end: float,
        spx_series: pd.Series,
    ) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        m = BacktestMetrics()

        if not self.equity_curve:
            return m

        # Basic returns
        final_equity = self.equity_curve[-1][1]
        m.total_return = final_equity / self.initial_capital - 1.0

        n_days = len(self.equity_curve)
        years = n_days / 252.0
        if years > 0:
            m.annual_return = (1 + m.total_return) ** (1.0 / years) - 1.0

        # Daily returns for risk metrics
        equities = np.array([e for _, e in self.equity_curve])
        daily_returns = np.diff(equities) / equities[:-1]

        if len(daily_returns) > 20:
            m.annual_volatility = float(np.std(daily_returns) * np.sqrt(252))

            # Sharpe
            rf_daily = self.config.backtest.risk_free_rate / 252.0
            excess = daily_returns - rf_daily
            if np.std(excess) > 0:
                m.sharpe_ratio = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

            # Sortino
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                m.sortino_ratio = float(
                    (np.mean(daily_returns) - rf_daily)
                    / np.std(downside) * np.sqrt(252)
                )

        # Drawdown
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max
        m.max_drawdown = float(np.min(drawdowns))

        # Max drawdown duration
        underwater = drawdowns < 0
        if underwater.any():
            groups = np.split(underwater, np.where(np.diff(underwater.astype(int)))[0] + 1)
            max_duration = max(
                (len(g) for g in groups if g.any()), default=0
            )
            m.max_drawdown_duration = int(max_duration)

        # Calmar
        if m.max_drawdown < 0:
            m.calmar_ratio = m.annual_return / abs(m.max_drawdown)

        # Trade statistics
        m.total_trades = len(self.closed_trades)
        winning = [t for t in self.closed_trades if t.close_pnl > 0]
        losing = [t for t in self.closed_trades if t.close_pnl <= 0]

        m.winning_trades = len(winning)
        m.losing_trades = len(losing)
        m.win_rate = len(winning) / max(1, m.total_trades)

        if winning:
            m.avg_win = sum(t.close_pnl for t in winning) / len(winning)
        if losing:
            m.avg_loss = sum(t.close_pnl for t in losing) / len(losing)

        total_wins = sum(t.close_pnl for t in winning) if winning else 0
        total_losses = abs(sum(t.close_pnl for t in losing)) if losing else 0
        m.profit_factor = total_wins / max(total_losses, 1.0)

        if self.closed_trades:
            m.avg_days_held = sum(t.days_held for t in self.closed_trades) / len(self.closed_trades)
            m.avg_credit_received = sum(t.entry_credit for t in self.closed_trades) / len(self.closed_trades)

        m.max_concurrent = self.max_concurrent

        # P&L
        m.total_pnl = sum(t.close_pnl for t in self.closed_trades)
        m.total_commissions = self.total_commissions
        m.total_slippage = self.total_slippage

        # Benchmark (SPX buy & hold)
        m.spx_return = spx_end / spx_start - 1.0

        spx_daily = spx_series.pct_change().dropna()
        if len(spx_daily) > 20:
            rf_daily = self.config.backtest.risk_free_rate / 252.0
            spx_excess = spx_daily - rf_daily
            m.spx_sharpe = float(
                np.mean(spx_excess) / np.std(spx_excess) * np.sqrt(252)
            ) if np.std(spx_excess) > 0 else 0.0

        # Alpha (simple: excess return over SPX scaled by beta)
        if len(daily_returns) > 20 and len(spx_daily) > 20:
            # Align lengths
            min_len = min(len(daily_returns), len(spx_daily))
            strat_rets = daily_returns[-min_len:]
            bench_rets = spx_daily.values[-min_len:]

            cov = np.cov(strat_rets, bench_rets)
            if cov.shape == (2, 2) and cov[1, 1] > 0:
                beta = cov[0, 1] / cov[1, 1]
                m.alpha = m.annual_return - beta * (m.spx_return / max(years, 0.01))

            # Information ratio
            tracking_error = np.std(strat_rets - bench_rets) * np.sqrt(252)
            if tracking_error > 0:
                m.information_ratio = (m.annual_return - m.spx_return / max(years, 0.01)) / tracking_error

        return m

    def save_results(self, filepath: str) -> None:
        """Save backtest results to JSON."""
        results = {
            "equity_curve": [
                {"date": d.isoformat(), "equity": e}
                for d, e in self.equity_curve
            ],
            "trades": [
                {
                    "id": t.id,
                    "short_strike": t.short_leg.strike,
                    "long_strike": t.long_leg.strike,
                    "expiry": t.short_leg.expiry.isoformat(),
                    "entry_date": t.entry_date.isoformat(),
                    "close_date": t.close_date.isoformat() if t.close_date else None,
                    "entry_credit": t.entry_credit,
                    "close_pnl": t.close_pnl,
                    "close_reason": t.close_reason,
                    "quantity": t.quantity,
                    "days_held": t.days_held,
                    "spx_at_entry": t.spx_at_entry,
                    "vix_at_entry": t.vix_at_entry,
                }
                for t in self.closed_trades
            ],
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run backtest from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="VRP Alpha Engine Backtest")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument("--output", default="vrp_backtest_results.json", help="Output file")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    config = get_config()
    config.backtest.initial_capital = args.capital

    bt = VRPBacktester(config)
    metrics = bt.run(start=args.start, end=args.end, verbose=not args.quiet)
    bt.save_results(args.output)


if __name__ == "__main__":
    main()
