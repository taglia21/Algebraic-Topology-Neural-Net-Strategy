"""
core/trade_journal.py
=====================
SQLite trade journal for persistent trade tracking and rolling Kelly parameter updates.
"""
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/trades.db")


class TradeJournal:
    def __init__(self, db_path: Path = _DEFAULT_DB_PATH):
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                fill_price REAL,
                order_type TEXT DEFAULT 'MKT',
                strategy_source TEXT DEFAULT 'TDA',
                signal_strength REAL,
                regime TEXT,
                status TEXT DEFAULT 'SUBMITTED'
            );

            CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT PRIMARY KEY,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                strategy_source TEXT DEFAULT 'TDA',
                stop_loss REAL,
                take_profit REAL
            );

            CREATE TABLE IF NOT EXISTS closed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                exit_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                strategy_source TEXT DEFAULT 'TDA',
                hold_days INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                nav REAL,
                daily_pnl REAL,
                positions_count INTEGER,
                trades_count INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL
            );
        """)
        self._conn.commit()

    def record_trade(self, ticker: str, action: str, quantity: float,
                     price: float = 0.0, fill_price: float = 0.0,
                     order_type: str = "MKT", strategy_source: str = "TDA",
                     signal_strength: float = 0.0, regime: str = "NORMAL",
                     status: str = "FILLED") -> int:
        """Record a trade execution."""
        cur = self._conn.execute(
            """INSERT INTO trades (timestamp, ticker, action, quantity, price,
               fill_price, order_type, strategy_source, signal_strength, regime, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), ticker, action, quantity, price,
             fill_price, order_type, strategy_source, signal_strength, regime, status)
        )
        self._conn.commit()

        # Track open position
        if action == "BUY" and status == "FILLED":
            self._open_position(ticker, fill_price or price, quantity, strategy_source)
        elif action == "SELL" and status == "FILLED":
            self._close_position(ticker, fill_price or price, quantity)

        return cur.lastrowid

    def _open_position(self, ticker: str, price: float, quantity: float,
                       strategy_source: str = "TDA") -> None:
        """Track an opened position."""
        existing = self._conn.execute(
            "SELECT * FROM positions WHERE ticker = ?", (ticker,)
        ).fetchone()

        if existing:
            # Average into existing position
            old_qty = existing["quantity"]
            old_price = existing["entry_price"]
            new_qty = old_qty + quantity
            avg_price = (old_price * old_qty + price * quantity) / new_qty if new_qty > 0 else price
            self._conn.execute(
                "UPDATE positions SET quantity = ?, entry_price = ? WHERE ticker = ?",
                (new_qty, avg_price, ticker)
            )
        else:
            self._conn.execute(
                """INSERT INTO positions (ticker, entry_date, entry_price, quantity, strategy_source)
                   VALUES (?, ?, ?, ?, ?)""",
                (ticker, datetime.now().strftime("%Y-%m-%d"), price, quantity, strategy_source)
            )
        self._conn.commit()

    def _close_position(self, ticker: str, exit_price: float, quantity: float) -> None:
        """Close (or reduce) a position and record P&L."""
        existing = self._conn.execute(
            "SELECT * FROM positions WHERE ticker = ?", (ticker,)
        ).fetchone()

        if not existing:
            logger.warning("No open position for %s to close", ticker)
            return

        entry_price = existing["entry_price"]
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

        entry_date = existing["entry_date"]
        exit_date = datetime.now().strftime("%Y-%m-%d")

        # Calculate hold days
        try:
            from datetime import datetime as dt
            ed = dt.strptime(entry_date, "%Y-%m-%d")
            xd = dt.strptime(exit_date, "%Y-%m-%d")
            hold_days = (xd - ed).days
        except Exception:
            hold_days = 0

        self._conn.execute(
            """INSERT INTO closed_trades (ticker, entry_date, exit_date, entry_price,
               exit_price, quantity, pnl, pnl_pct, strategy_source, hold_days)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, entry_date, exit_date, entry_price, exit_price,
             quantity, pnl, pnl_pct, existing["strategy_source"], hold_days)
        )

        remaining = existing["quantity"] - quantity
        if remaining <= 0.001:  # fully closed
            self._conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
        else:
            self._conn.execute(
                "UPDATE positions SET quantity = ? WHERE ticker = ?",
                (remaining, ticker)
            )
        self._conn.commit()

    def get_kelly_params(self, lookback: int = 100) -> Dict[str, float]:
        """Compute rolling Kelly parameters from closed trades.

        Returns dict with win_rate, avg_win, avg_loss.
        Falls back to conservative defaults if insufficient data.
        """
        rows = self._conn.execute(
            "SELECT pnl_pct FROM closed_trades ORDER BY id DESC LIMIT ?",
            (lookback,)
        ).fetchall()

        if len(rows) < 10:  # Need minimum sample
            return {"win_rate": 0.55, "avg_win": 0.02, "avg_loss": 0.015}

        pnls = [r["pnl_pct"] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.5
        avg_win = sum(wins) / len(wins) if wins else 0.02
        avg_loss = sum(losses) / len(losses) if losses else 0.015

        # Safety clamp
        win_rate = max(0.30, min(0.80, win_rate))
        avg_win = max(0.005, min(0.10, avg_win))
        avg_loss = max(0.005, min(0.10, avg_loss))

        return {"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss}

    def record_daily_stats(self, nav: float, daily_pnl: float,
                           positions_count: int, trades_count: int) -> None:
        """Record end-of-day statistics."""
        kelly = self.get_kelly_params()
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._conn.execute(
            """INSERT OR REPLACE INTO daily_stats
               (date, nav, daily_pnl, positions_count, trades_count, win_rate, avg_win, avg_loss)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, nav, daily_pnl, positions_count, trades_count,
             kelly["win_rate"], kelly["avg_win"], kelly["avg_loss"])
        )
        self._conn.commit()

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trade history."""
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        closed = self._conn.execute(
            "SELECT COUNT(*) as total, SUM(pnl) as total_pnl FROM closed_trades"
        ).fetchone()

        kelly = self.get_kelly_params()

        return {
            "total_closed_trades": closed["total"] or 0,
            "total_pnl": closed["total_pnl"] or 0.0,
            "win_rate": kelly["win_rate"],
            "avg_win": kelly["avg_win"],
            "avg_loss": kelly["avg_loss"],
        }

    def get_stale_positions(self, max_age_days: int = 5, min_gain_pct: float = 0.01) -> List[Dict]:
        """Find positions held too long with insufficient gains.

        Returns positions where hold_days > max_age_days and unrealized gain < min_gain_pct.
        These are candidates for reduction/exit.
        """
        positions = self.get_open_positions()
        stale = []
        for pos in positions:
            try:
                from datetime import datetime as dt
                entry = dt.strptime(pos["entry_date"], "%Y-%m-%d")
                age = (dt.now() - entry).days
                if age > max_age_days:
                    stale.append({**pos, "hold_days": age})
            except Exception:
                pass
        return stale

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
