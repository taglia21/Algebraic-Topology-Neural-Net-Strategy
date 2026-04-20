"""
core/trade_journal.py
=====================
Persistent trade journal with online learning feedback loop.

Tracks every trade, computes rolling Kelly parameters, signal accuracy,
and provides real-time feedback to improve signal quality thresholds.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JOURNAL_DIR = Path(os.environ.get("ATNN_DATA_DIR", "/app/data")) / "journal"


@dataclass
class TradeRecord:
    """Single trade record."""
    trade_id: str
    ticker: str
    action: str            # BUY, SELL
    quantity: float
    entry_price: float
    fill_price: float
    strategy_source: str   # TDA, NN, ENSEMBLE, INTRADAY_STOP, INTRADAY_PROFIT
    signal_strength: float
    regime: str
    timestamp: str
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_minutes: Optional[float] = None
    closed: bool = False


@dataclass
class DailyStats:
    """End-of-day stats."""
    date: str
    nav: float
    daily_pnl: float
    position_count: int
    signals_generated: int
    trades_executed: int
    win_count: int
    loss_count: int


class TradeJournal:
    """Persistent trade journal with online learning.

    Saves trades to JSON files for durability across restarts.
    Computes rolling statistics for Kelly criterion and
    adaptive signal threshold tuning.
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._dir = Path(data_dir) if data_dir else _JOURNAL_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        self._trades_file = self._dir / "trades.jsonl"
        self._daily_file = self._dir / "daily_stats.jsonl"
        self._params_file = self._dir / "learned_params.json"

        # In-memory state
        self._trades: List[TradeRecord] = []
        self._daily_stats: List[DailyStats] = []
        self._learned_params: Dict[str, Any] = {}

        # Load existing data
        self._load()
        logger.info(
            "TradeJournal: %d trades, %d daily records loaded",
            len(self._trades), len(self._daily_stats),
        )

    def _load(self) -> None:
        """Load persisted trades and stats."""
        if self._trades_file.exists():
            try:
                with open(self._trades_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            d = json.loads(line)
                            self._trades.append(TradeRecord(**d))
            except Exception as e:
                logger.warning("Failed to load trades: %s", e)

        if self._daily_file.exists():
            try:
                with open(self._daily_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            d = json.loads(line)
                            self._daily_stats.append(DailyStats(**d))
            except Exception as e:
                logger.warning("Failed to load daily stats: %s", e)

        if self._params_file.exists():
            try:
                self._learned_params = json.loads(self._params_file.read_text())
            except Exception as e:
                logger.warning("Failed to load learned params: %s", e)

    def _append_trade(self, trade: TradeRecord) -> None:
        """Append a trade to the JSONL file."""
        try:
            with open(self._trades_file, "a") as f:
                f.write(json.dumps(asdict(trade)) + "\n")
        except Exception as e:
            logger.warning("Failed to persist trade: %s", e)

    def _rewrite_trades(self) -> None:
        """Rewrite the full trades file (used after updating records)."""
        try:
            with open(self._trades_file, "w") as f:
                for t in self._trades:
                    f.write(json.dumps(asdict(t)) + "\n")
        except Exception as e:
            logger.warning("Failed to rewrite trades file: %s", e)

    def _save_params(self) -> None:
        """Persist learned parameters."""
        try:
            self._params_file.write_text(json.dumps(self._learned_params, indent=2))
        except Exception as e:
            logger.warning("Failed to save learned params: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(
        self,
        ticker: str,
        action: str,
        quantity: float,
        price: float,
        fill_price: float,
        strategy_source: str = "ENSEMBLE",
        signal_strength: float = 0.0,
        regime: str = "NORMAL",
    ) -> TradeRecord:
        """Record a new trade entry or exit."""
        trade_id = f"{ticker}_{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trade = TradeRecord(
            trade_id=trade_id,
            ticker=ticker,
            action=action,
            quantity=quantity,
            entry_price=price,
            fill_price=fill_price,
            strategy_source=strategy_source,
            signal_strength=signal_strength,
            regime=regime,
            timestamp=datetime.now().isoformat(),
        )
        self._trades.append(trade)
        self._append_trade(trade)

        # If this is a SELL, try to match with open BUY and compute P&L
        if action == "SELL":
            self._close_matching_buy(ticker, fill_price, quantity)

        # Update learned params after every trade
        self._update_learned_params()

        return trade

    def _close_matching_trade(
        self, ticker: str, exit_price: float, quantity: float, exit_action: str = "SELL"
    ) -> None:
        """Match an exit with the oldest open entry for the same ticker.

        MEDIUM-02 FIX: Handles both LONG (BUY→SELL) and SHORT (SELL→BUY) trades.
        """
        # Determine which entry action to match
        if exit_action == "SELL":
            entry_action = "BUY"   # Closing a long
        else:
            entry_action = "SELL"  # Covering a short

        for trade in self._trades:
            if (
                trade.ticker == ticker
                and trade.action == entry_action
                and not trade.closed
            ):
                trade.exit_price = exit_price
                trade.exit_timestamp = datetime.now().isoformat()

                if entry_action == "BUY":
                    # Long trade: profit = exit - entry
                    trade.pnl = (exit_price - trade.fill_price) * trade.quantity
                else:
                    # Short trade: profit = entry - exit (inverted)
                    trade.pnl = (trade.fill_price - exit_price) * trade.quantity

                trade.pnl_pct = (
                    trade.pnl / (trade.fill_price * trade.quantity)
                    if trade.fill_price > 0
                    else 0.0
                )
                entry_time = datetime.fromisoformat(trade.timestamp)
                trade.hold_minutes = (
                    (datetime.now() - entry_time).total_seconds() / 60.0
                )
                trade.closed = True
                logger.info(
                    "Closed %s trade %s: P&L=$%.2f (%.2f%%) held %.0f min",
                    "LONG" if entry_action == "BUY" else "SHORT",
                    trade.trade_id, trade.pnl, trade.pnl_pct * 100,
                    trade.hold_minutes,
                )
                break

        self._rewrite_trades()

    # Backward-compatible alias
    def _close_matching_buy(self, ticker, exit_price, quantity):
        return self._close_matching_trade(ticker, exit_price, quantity, "SELL")

    def record_daily_stats(
        self,
        nav: float,
        daily_pnl: float,
        position_count: int,
        signals_generated: int,
    ) -> None:
        """Record end-of-day statistics."""
        today = datetime.now().strftime("%Y-%m-%d")
        closed_today = [
            t for t in self._trades
            if t.closed and t.exit_timestamp and t.exit_timestamp.startswith(today)
        ]
        wins = sum(1 for t in closed_today if t.pnl and t.pnl > 0)
        losses = sum(1 for t in closed_today if t.pnl and t.pnl <= 0)

        stats = DailyStats(
            date=today,
            nav=nav,
            daily_pnl=daily_pnl,
            position_count=position_count,
            signals_generated=signals_generated,
            trades_executed=len(closed_today),
            win_count=wins,
            loss_count=losses,
        )
        self._daily_stats.append(stats)
        try:
            with open(self._daily_file, "a") as f:
                f.write(json.dumps(asdict(stats)) + "\n")
        except Exception as e:
            logger.warning("Failed to persist daily stats: %s", e)

    def get_kelly_params(
        self, lookback: int = 50
    ) -> Dict[str, float]:
        """Compute rolling Kelly parameters from recent closed trades.

        Returns dict with win_rate, avg_win, avg_loss.
        Falls back to conservative defaults if insufficient data.
        """
        closed = [t for t in self._trades if t.closed and t.pnl is not None]
        recent = closed[-lookback:] if len(closed) > lookback else closed

        if len(recent) < 5:
            # Not enough data — use conservative defaults
            return {"win_rate": 0.52, "avg_win": 0.015, "avg_loss": 0.012}

        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl <= 0]

        win_rate = len(wins) / len(recent) if recent else 0.5
        avg_win = (
            sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.015
        )
        avg_loss = (
            abs(sum(t.pnl_pct for t in losses) / len(losses))
            if losses else 0.012
        )

        # Clamp to reasonable ranges
        win_rate = max(0.30, min(0.80, win_rate))
        avg_win = max(0.002, min(0.10, avg_win))
        avg_loss = max(0.002, min(0.10, avg_loss))

        return {"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss}

    def get_open_positions(self) -> List[Dict]:
        """Get currently open (unclosed) buy trades."""
        return [
            asdict(t) for t in self._trades
            if t.action == "BUY" and not t.closed
        ]

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get most recent trades."""
        return [asdict(t) for t in self._trades[-limit:]]

    def get_learned_params(self) -> Dict[str, Any]:
        """Get current learned parameters for adaptive thresholds."""
        return dict(self._learned_params)

    def _update_learned_params(self) -> None:
        """Online learning: update parameters based on trade outcomes.

        Adapts:
        - optimal_min_signal_strength: threshold that maximizes hit rate
        - per-ticker performance: which tickers the model trades well
        - regime_performance: how the model performs in each regime
        """
        closed = [t for t in self._trades if t.closed and t.pnl is not None]
        if len(closed) < 10:
            return

        # --- Optimal signal strength threshold ---
        # Find the threshold that maximizes profit factor
        strengths = sorted(set(t.signal_strength for t in closed if t.signal_strength > 0))
        best_threshold = 0.15
        best_profit_factor = 0.0

        for threshold in strengths:
            above = [t for t in closed if t.signal_strength >= threshold]
            if len(above) < 5:
                continue
            gross_profit = sum(t.pnl for t in above if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in above if t.pnl < 0)) or 1.0
            pf = gross_profit / gross_loss
            if pf > best_profit_factor:
                best_profit_factor = pf
                best_threshold = threshold

        self._learned_params["optimal_min_signal_strength"] = round(
            best_threshold, 4
        )
        self._learned_params["profit_factor"] = round(best_profit_factor, 4)

        # --- Per-ticker hit rate ---
        ticker_stats: Dict[str, Dict] = {}
        for t in closed:
            if t.ticker not in ticker_stats:
                ticker_stats[t.ticker] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
            if t.pnl > 0:
                ticker_stats[t.ticker]["wins"] += 1
            else:
                ticker_stats[t.ticker]["losses"] += 1
            ticker_stats[t.ticker]["total_pnl"] += t.pnl

        self._learned_params["ticker_performance"] = {
            ticker: {
                "win_rate": s["wins"] / (s["wins"] + s["losses"])
                if (s["wins"] + s["losses"]) > 0 else 0.5,
                "total_pnl": round(s["total_pnl"], 2),
                "trade_count": s["wins"] + s["losses"],
            }
            for ticker, s in ticker_stats.items()
        }

        # --- Regime performance ---
        regime_stats: Dict[str, Dict] = {}
        for t in closed:
            r = t.regime or "NORMAL"
            if r not in regime_stats:
                regime_stats[r] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
            if t.pnl > 0:
                regime_stats[r]["wins"] += 1
            else:
                regime_stats[r]["losses"] += 1
            regime_stats[r]["total_pnl"] += t.pnl

        self._learned_params["regime_performance"] = {
            regime: {
                "win_rate": s["wins"] / (s["wins"] + s["losses"])
                if (s["wins"] + s["losses"]) > 0 else 0.5,
                "total_pnl": round(s["total_pnl"], 2),
            }
            for regime, s in regime_stats.items()
        }

        self._save_params()
        logger.info(
            "Online learning update: optimal_threshold=%.4f, profit_factor=%.2f, "
            "%d tickers tracked",
            best_threshold, best_profit_factor, len(ticker_stats),
        )

    def get_ticker_blacklist(self, min_trades: int = 5, max_loss_rate: float = 0.70) -> List[str]:
        """Return tickers that consistently lose money.

        These should be skipped or given reduced weight.
        """
        perf = self._learned_params.get("ticker_performance", {})
        blacklist = []
        for ticker, stats in perf.items():
            if stats["trade_count"] >= min_trades and stats["win_rate"] < (1 - max_loss_rate):
                blacklist.append(ticker)
        return blacklist

    def close(self) -> None:
        """Flush and close the journal."""
        self._rewrite_trades()
        self._save_params()
        logger.info("TradeJournal closed")
