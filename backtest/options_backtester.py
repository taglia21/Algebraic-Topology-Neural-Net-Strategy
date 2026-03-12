"""
backtest/options_backtester.py
==============================
Options-specific backtester for the ATNN $444 account.

Black-Scholes pricing, Greeks computation, defined-risk strategies
(verticals, iron condors, single legs), daily mark-to-market,
and Greeks P&L attribution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_DEFAULT_RISK_FREE_RATE: float = 0.05
_DEFAULT_MAX_RISK_PER_TRADE: float = 50.0   # $50
_DEFAULT_MAX_POSITIONS: int = 3
_OPTION_COMMISSION_PER_CONTRACT: float = 0.65
_OPTION_COMMISSION_MIN: float = 1.00
_DEFAULT_SLIPPAGE_PCT: float = 0.02  # 2% of option price


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

class BlackScholes:
    """Analytical Black-Scholes pricing and Greeks."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0.0)
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        d_2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d_1) - K * math.exp(-r * T) * norm.cdf(d_2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S, 0.0)
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        d_2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
        if T <= 0 or sigma <= 0:
            if is_call:
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        if is_call:
            return norm.cdf(d_1)
        return norm.cdf(d_1) - 1.0

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d_1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
        """Daily theta (per calendar day, i.e. divided by 365)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        d_2 = BlackScholes.d2(S, K, T, r, sigma)
        term1 = -(S * norm.pdf(d_1) * sigma) / (2 * math.sqrt(T))
        if is_call:
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d_2)
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d_2)
        return (term1 + term2) / 365.0

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Vega per 1% move in IV (i.e. / 100)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d_1) * math.sqrt(T) / 100.0

    @staticmethod
    def price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
        if is_call:
            return BlackScholes.call_price(S, K, T, r, sigma)
        return BlackScholes.put_price(S, K, T, r, sigma)


# ---------------------------------------------------------------------------
# Option leg / position
# ---------------------------------------------------------------------------

@dataclass
class OptionLeg:
    """A single option leg."""
    strike: float
    is_call: bool
    is_long: bool  # True = bought, False = sold/written
    quantity: int = 1
    entry_price: float = 0.0  # per-contract premium


@dataclass
class OptionPosition:
    """An options position (one or more legs forming a strategy)."""
    position_id: int
    symbol: str
    strategy_type: str  # "vertical", "iron_condor", "single"
    legs: List[OptionLeg]
    entry_date: pd.Timestamp
    expiration_dte: int  # days to expiration at entry
    entry_cost: float  # net debit/credit (positive = debit)
    max_profit: float
    max_loss: float  # always positive
    current_value: float = 0.0
    closed: bool = False
    exit_date: Optional[pd.Timestamp] = None
    exit_value: float = 0.0
    pnl: float = 0.0
    # Greeks attribution
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0


@dataclass
class OptionsBacktestResult:
    """Result from an options backtest."""
    equity_curve: pd.Series
    options_trades: pd.DataFrame
    greeks_pnl_attribution: Dict[str, float]
    strategy_breakdown: Dict[str, dict]
    metrics: dict
    initial_capital: float = 444.0


# ---------------------------------------------------------------------------
# OptionsBacktester
# ---------------------------------------------------------------------------

class OptionsBacktester:
    """Options-specific backtester for the $444 account.

    Parameters
    ----------
    max_risk_per_trade : float
        Maximum dollar risk per trade (default $50).
    max_positions : int
        Maximum concurrent open positions (default 3).
    target_profit_pct : float
        Take profit at this % of max profit (default 50%).
    close_dte : int
        Close positions when DTE <= this (default 2).
    slippage_pct : float
        Slippage as % of option price (default 2%).
    risk_free_rate : float
        Annual risk-free rate for BS pricing.
    """

    def __init__(
        self,
        max_risk_per_trade: float = _DEFAULT_MAX_RISK_PER_TRADE,
        max_positions: int = _DEFAULT_MAX_POSITIONS,
        target_profit_pct: float = 0.50,
        close_dte: int = 2,
        slippage_pct: float = _DEFAULT_SLIPPAGE_PCT,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.target_profit_pct = target_profit_pct
        self.close_dte = close_dte
        self.slippage_pct = slippage_pct
        self.risk_free_rate = risk_free_rate

        # State
        self._cash: float = 0.0
        self._positions: List[OptionPosition] = []
        self._closed_trades: List[OptionPosition] = []
        self._equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self._next_id: int = 1

    def _reset(self, initial_capital: float) -> None:
        self._cash = initial_capital
        self._positions = []
        self._closed_trades = []
        self._equity_curve = []
        self._next_id = 1

    # ----- Position value & Greeks -------------------------------------------

    def _value_position(
        self,
        pos: OptionPosition,
        underlying_price: float,
        dte: int,
        iv: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Mark-to-market a position using Black-Scholes.

        Returns (net_value, {delta, gamma, theta, vega}).
        """
        T = max(dte / 365.0, 1e-6)
        net_value = 0.0
        net_greeks: Dict[str, float] = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        for leg in pos.legs:
            price = BlackScholes.price(underlying_price, leg.strike, T, self.risk_free_rate, iv, leg.is_call)
            d = BlackScholes.delta(underlying_price, leg.strike, T, self.risk_free_rate, iv, leg.is_call)
            g = BlackScholes.gamma(underlying_price, leg.strike, T, self.risk_free_rate, iv)
            th = BlackScholes.theta(underlying_price, leg.strike, T, self.risk_free_rate, iv, leg.is_call)
            v = BlackScholes.vega(underlying_price, leg.strike, T, self.risk_free_rate, iv)

            multiplier = leg.quantity * (1 if leg.is_long else -1) * 100  # 100 shares per contract

            net_value += price * multiplier
            net_greeks["delta"] += d * multiplier / 100
            net_greeks["gamma"] += g * multiplier / 100
            net_greeks["theta"] += th * multiplier
            net_greeks["vega"] += v * multiplier

        return net_value, net_greeks

    # ----- Commission --------------------------------------------------------

    @staticmethod
    def _commission(legs: List[OptionLeg]) -> float:
        """Compute round-trip commission for option legs."""
        total_contracts = sum(leg.quantity for leg in legs)
        return max(_OPTION_COMMISSION_MIN, total_contracts * _OPTION_COMMISSION_PER_CONTRACT)

    # ----- Strategy builders -------------------------------------------------

    def _build_vertical_spread(
        self,
        symbol: str,
        underlying_price: float,
        iv: float,
        dte: int,
        is_bullish: bool,
        use_calls: bool,
        date: pd.Timestamp,
    ) -> Optional[OptionPosition]:
        """Build a vertical spread within max_risk constraint.

        Bull call spread: buy lower strike call, sell higher strike call.
        Bear put spread: buy higher strike put, sell lower strike put.
        Bull put spread: sell higher strike put, buy lower strike put.
        Bear call spread: sell lower strike call, buy higher strike call.
        """
        T = dte / 365.0
        # Choose strikes based on ~0.30-0.40 delta
        strike_step = max(1.0, round(underlying_price * 0.02))  # ~2% OTM

        if is_bullish and use_calls:
            # Bull call spread (debit)
            long_strike = round(underlying_price / strike_step) * strike_step
            short_strike = long_strike + strike_step
            long_leg = OptionLeg(strike=long_strike, is_call=True, is_long=True)
            short_leg = OptionLeg(strike=short_strike, is_call=True, is_long=False)
        elif not is_bullish and use_calls:
            # Bear call spread (credit)
            short_strike = round(underlying_price / strike_step) * strike_step
            long_strike = short_strike + strike_step
            long_leg = OptionLeg(strike=long_strike, is_call=True, is_long=True)
            short_leg = OptionLeg(strike=short_strike, is_call=True, is_long=False)
        elif is_bullish and not use_calls:
            # Bull put spread (credit)
            short_strike = round(underlying_price / strike_step) * strike_step
            long_strike = short_strike - strike_step
            long_leg = OptionLeg(strike=long_strike, is_call=False, is_long=True)
            short_leg = OptionLeg(strike=short_strike, is_call=False, is_long=False)
        else:
            # Bear put spread (debit)
            long_strike = round(underlying_price / strike_step) * strike_step
            short_strike = long_strike - strike_step
            long_leg = OptionLeg(strike=long_strike, is_call=False, is_long=True)
            short_leg = OptionLeg(strike=short_strike, is_call=False, is_long=False)

        legs = [long_leg, short_leg]

        # Price the spread
        long_price = BlackScholes.price(underlying_price, long_leg.strike, T, self.risk_free_rate, iv, long_leg.is_call)
        short_price = BlackScholes.price(underlying_price, short_leg.strike, T, self.risk_free_rate, iv, short_leg.is_call)

        net_debit = (long_price - short_price) * 100  # per contract, *100 shares
        spread_width = abs(long_leg.strike - short_leg.strike) * 100

        if net_debit > 0:
            # Debit spread
            max_loss = net_debit
            max_profit = spread_width - net_debit
        else:
            # Credit spread
            credit = abs(net_debit)
            max_loss = spread_width - credit
            max_profit = credit

        # Check risk constraint
        if max_loss > self.max_risk_per_trade:
            return None

        # Apply slippage
        slippage = abs(net_debit) * self.slippage_pct
        entry_cost = net_debit + slippage if net_debit > 0 else net_debit - slippage

        # Set entry prices on legs
        long_leg.entry_price = long_price
        short_leg.entry_price = short_price

        return OptionPosition(
            position_id=self._next_id,
            symbol=symbol,
            strategy_type="vertical",
            legs=legs,
            entry_date=date,
            expiration_dte=dte,
            entry_cost=entry_cost,
            max_profit=max_profit,
            max_loss=max_loss,
            current_value=-entry_cost,  # If debit, we paid, so value starts negative of cost
        )

    def _build_iron_condor(
        self,
        symbol: str,
        underlying_price: float,
        iv: float,
        dte: int,
        date: pd.Timestamp,
    ) -> Optional[OptionPosition]:
        """Build an iron condor (credit strategy).

        Sell call + sell put OTM, buy further OTM call + put for protection.
        """
        T = dte / 365.0
        step = max(1.0, round(underlying_price * 0.03))  # ~3% OTM for short strikes
        wing = max(1.0, round(underlying_price * 0.02))  # 2% wide wings

        short_call_strike = round(underlying_price / step) * step + step
        long_call_strike = short_call_strike + wing
        short_put_strike = round(underlying_price / step) * step - step
        long_put_strike = short_put_strike - wing

        legs = [
            OptionLeg(strike=long_put_strike, is_call=False, is_long=True),
            OptionLeg(strike=short_put_strike, is_call=False, is_long=False),
            OptionLeg(strike=short_call_strike, is_call=True, is_long=False),
            OptionLeg(strike=long_call_strike, is_call=True, is_long=True),
        ]

        # Price each leg
        prices = []
        for leg in legs:
            p = BlackScholes.price(underlying_price, leg.strike, T, self.risk_free_rate, iv, leg.is_call)
            leg.entry_price = p
            prices.append(p)

        # Net credit = sold premiums - bought premiums
        net_credit = (prices[1] + prices[2] - prices[0] - prices[3]) * 100
        wing_width = wing * 100
        max_loss = wing_width - net_credit
        max_profit = net_credit

        if max_loss > self.max_risk_per_trade or net_credit <= 0:
            return None

        slippage = net_credit * self.slippage_pct
        entry_cost = -(net_credit - slippage)  # Negative = credit received

        return OptionPosition(
            position_id=self._next_id,
            symbol=symbol,
            strategy_type="iron_condor",
            legs=legs,
            entry_date=date,
            expiration_dte=dte,
            entry_cost=entry_cost,
            max_profit=max_profit,
            max_loss=max_loss,
            current_value=-entry_cost,
        )

    def _build_single_leg(
        self,
        symbol: str,
        underlying_price: float,
        iv: float,
        dte: int,
        is_call: bool,
        date: pd.Timestamp,
    ) -> Optional[OptionPosition]:
        """Build a single directional long option."""
        T = dte / 365.0
        # Slightly OTM
        step = max(1.0, round(underlying_price * 0.02))
        if is_call:
            strike = round(underlying_price / step) * step + step
        else:
            strike = round(underlying_price / step) * step - step

        price = BlackScholes.price(underlying_price, strike, T, self.risk_free_rate, iv, is_call)
        cost = price * 100  # per contract

        if cost > self.max_risk_per_trade or cost <= 0:
            return None

        leg = OptionLeg(strike=strike, is_call=is_call, is_long=True, entry_price=price)
        slippage = cost * self.slippage_pct

        return OptionPosition(
            position_id=self._next_id,
            symbol=symbol,
            strategy_type="single",
            legs=[leg],
            entry_date=date,
            expiration_dte=dte,
            entry_cost=cost + slippage,
            max_profit=float("inf"),  # Uncapped for long options
            max_loss=cost + slippage,
            current_value=-cost - slippage,
        )

    # ----- Main run ----------------------------------------------------------

    def run(
        self,
        signals: pd.DataFrame,
        underlying_prices: pd.DataFrame,
        vol_surface: Optional[pd.DataFrame] = None,
        initial_capital: float = 444.0,
        default_iv: float = 0.25,
        default_dte: int = 30,
    ) -> OptionsBacktestResult:
        """Run the options backtest.

        Parameters
        ----------
        signals : pd.DataFrame
            Indexed by date. Columns: symbol (str), direction (1/-1),
            strategy_type (vertical/iron_condor/single), strength (0-1).
            OR: columns are symbols with direction values.
        underlying_prices : pd.DataFrame
            Indexed by date, columns include 'close' (or symbol names).
        vol_surface : pd.DataFrame, optional
            Indexed by date, columns = symbol, values = implied volatility.
            If None, uses default_iv.
        initial_capital : float
            Starting capital (default $444).
        default_iv : float
            Default implied volatility if vol_surface not provided.
        default_dte : int
            Default days to expiration for new trades.

        Returns
        -------
        OptionsBacktestResult
        """
        self._reset(initial_capital)

        dates = sorted(underlying_prices.index)
        if not dates:
            return self._build_result(initial_capital)

        # Determine if prices are single-column or multi-symbol
        price_col = None
        for c in ["close", "Close", "adj_close"]:
            if c in underlying_prices.columns:
                price_col = c
                break

        for i, date in enumerate(dates):
            # Get underlying price
            if price_col:
                spot = float(underlying_prices.loc[date, price_col])
                symbol = "UNDERLYING"
            else:
                # Multi-symbol: take first column as underlying
                spot = float(underlying_prices.iloc[i, 0])
                symbol = str(underlying_prices.columns[0])

            if spot <= 0:
                continue

            # Get IV for today
            iv = default_iv
            if vol_surface is not None and date in vol_surface.index:
                iv_row = vol_surface.loc[date]
                if isinstance(iv_row, pd.Series) and len(iv_row) > 0:
                    iv = float(iv_row.iloc[0])
                elif isinstance(iv_row, (int, float)):
                    iv = float(iv_row)

            # 1. Mark-to-market existing positions & check exit conditions
            days_elapsed = (i > 0)
            self._manage_positions(date, spot, iv, default_dte, i)

            # 2. Process new signals
            if date in signals.index:
                sig = signals.loc[date]
                if isinstance(sig, pd.DataFrame):
                    sig = sig.iloc[0]

                self._process_signal(sig, symbol, spot, iv, default_dte, date)

            # 3. Record equity
            nav = self._compute_nav(spot, iv, default_dte, i)
            self._equity_curve.append((date, nav))

        # Force-close remaining positions
        if dates and self._positions:
            last_spot = float(underlying_prices.loc[dates[-1], price_col]) if price_col else float(underlying_prices.iloc[-1, 0])
            for pos in list(self._positions):
                self._close_position(pos, dates[-1], last_spot, default_iv, 0)

        return self._build_result(initial_capital)

    def _process_signal(
        self,
        sig: pd.Series,
        symbol: str,
        spot: float,
        iv: float,
        dte: int,
        date: pd.Timestamp,
    ) -> None:
        """Process a signal row and open positions if appropriate."""
        # Check position limit
        open_count = len(self._positions)
        if open_count >= self.max_positions:
            return

        # Parse signal
        direction = 0
        strategy_type = "vertical"

        if "direction" in sig.index:
            direction = int(sig["direction"]) if not pd.isna(sig.get("direction", 0)) else 0
        elif len(sig) > 0:
            # Columns-as-symbols format: take first non-zero
            for col_val in sig:
                if isinstance(col_val, (int, float)) and col_val != 0:
                    direction = 1 if col_val > 0 else -1
                    break

        if direction == 0:
            return

        if "strategy_type" in sig.index and not pd.isna(sig.get("strategy_type")):
            strategy_type = str(sig["strategy_type"]).lower()

        # Build position
        pos: Optional[OptionPosition] = None
        is_bullish = direction > 0

        if strategy_type == "iron_condor":
            pos = self._build_iron_condor(symbol, spot, iv, dte, date)
        elif strategy_type == "single":
            pos = self._build_single_leg(symbol, spot, iv, dte, is_call=is_bullish, date=date)
        else:
            # Default: vertical spread
            pos = self._build_vertical_spread(
                symbol, spot, iv, dte, is_bullish=is_bullish,
                use_calls=is_bullish, date=date,
            )

        if pos is None:
            return

        # Check we can afford it
        cost = max(0, pos.entry_cost)
        commission = self._commission(pos.legs)
        total_cost = cost + commission

        if total_cost > self._cash:
            return

        self._cash -= total_cost
        self._next_id += 1
        self._positions.append(pos)
        logger.debug(
            "Opened %s %s on %s: cost=$%.2f, max_loss=$%.2f",
            pos.strategy_type, symbol, date, total_cost, pos.max_loss,
        )

    def _manage_positions(
        self,
        date: pd.Timestamp,
        spot: float,
        iv: float,
        original_dte: int,
        bar_idx: int,
    ) -> None:
        """Check all open positions for exit conditions."""
        remaining = []
        for pos in self._positions:
            # Calculate remaining DTE
            days_held = max(1, (date - pos.entry_date).days)
            remaining_dte = max(0, pos.expiration_dte - days_held)

            # Mark-to-market
            value, greeks = self._value_position(pos, spot, remaining_dte, iv)

            # Greeks P&L attribution (daily)
            if bar_idx > 0:
                pos.theta_pnl += greeks.get("theta", 0)
                # Delta/gamma/vega PnL would need previous price — approximate
                pos.delta_pnl += greeks.get("delta", 0) * 0  # Placeholder
                pos.vega_pnl += greeks.get("vega", 0) * 0

            pos.current_value = value
            current_pnl = value + pos.entry_cost  # entry_cost is negative for credits

            # Exit conditions
            should_close = False
            reason = ""

            # 1. Target profit
            if pos.max_profit > 0 and current_pnl >= pos.max_profit * self.target_profit_pct:
                should_close = True
                reason = "target_profit"

            # 2. Max loss
            if current_pnl <= -pos.max_loss:
                should_close = True
                reason = "max_loss"

            # 3. Near expiration (gamma risk)
            if remaining_dte <= self.close_dte:
                should_close = True
                reason = "near_expiration"

            # 4. Expired
            if remaining_dte <= 0:
                should_close = True
                reason = "expired"

            if should_close:
                self._close_position(pos, date, spot, iv, remaining_dte)
            else:
                remaining.append(pos)

        self._positions = remaining

    def _close_position(
        self,
        pos: OptionPosition,
        date: pd.Timestamp,
        spot: float,
        iv: float,
        remaining_dte: int,
    ) -> None:
        """Close an options position."""
        # Mark final value
        if remaining_dte > 0:
            value, _ = self._value_position(pos, spot, remaining_dte, iv)
        else:
            # At expiration: intrinsic value only
            value = 0.0
            for leg in pos.legs:
                if leg.is_call:
                    intrinsic = max(spot - leg.strike, 0) * 100
                else:
                    intrinsic = max(leg.strike - spot, 0) * 100
                multiplier = leg.quantity * (1 if leg.is_long else -1)
                value += intrinsic * multiplier

        # Apply closing slippage
        slippage = abs(value) * self.slippage_pct
        exit_value = value - slippage if value > 0 else value + slippage

        commission = self._commission(pos.legs)
        pos.exit_value = exit_value
        pos.pnl = exit_value + pos.entry_cost - commission  # entry_cost < 0 for credits
        pos.exit_date = date
        pos.closed = True

        self._cash += exit_value - commission
        self._closed_trades.append(pos)

    def _compute_nav(
        self,
        spot: float,
        iv: float,
        original_dte: int,
        bar_idx: int,
    ) -> float:
        """Compute NAV = cash + sum of position values."""
        nav = self._cash
        for pos in self._positions:
            nav += pos.current_value
        return nav

    def _build_result(self, initial_capital: float) -> OptionsBacktestResult:
        """Assemble the OptionsBacktestResult."""
        from backtest.metrics import BacktestMetrics

        if self._equity_curve:
            dates, values = zip(*self._equity_curve)
            eq = pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")
        else:
            eq = pd.Series(dtype=float, name="equity")

        # Build trades DataFrame
        records = []
        for pos in self._closed_trades:
            records.append({
                "symbol": pos.symbol,
                "strategy_type": pos.strategy_type,
                "entry_date": pos.entry_date,
                "exit_date": pos.exit_date,
                "entry_cost": pos.entry_cost,
                "exit_value": pos.exit_value,
                "max_profit": pos.max_profit,
                "max_loss": pos.max_loss,
                "pnl": pos.pnl,
                "holding_days": (pos.exit_date - pos.entry_date).days if pos.exit_date else 0,
                "n_legs": len(pos.legs),
                "delta_pnl": pos.delta_pnl,
                "gamma_pnl": pos.gamma_pnl,
                "theta_pnl": pos.theta_pnl,
                "vega_pnl": pos.vega_pnl,
                "asset_type": "OPTION",
            })

        trades_df = pd.DataFrame(records) if records else pd.DataFrame(columns=[
            "symbol", "strategy_type", "entry_date", "exit_date", "entry_cost",
            "exit_value", "max_profit", "max_loss", "pnl", "holding_days",
            "n_legs", "delta_pnl", "gamma_pnl", "theta_pnl", "vega_pnl", "asset_type",
        ])

        # Greeks P&L attribution
        greeks_attr = {
            "total_delta_pnl": sum(r.get("delta_pnl", 0) for r in records),
            "total_gamma_pnl": sum(r.get("gamma_pnl", 0) for r in records),
            "total_theta_pnl": sum(r.get("theta_pnl", 0) for r in records),
            "total_vega_pnl": sum(r.get("vega_pnl", 0) for r in records),
        }

        # Strategy breakdown
        strat_breakdown: Dict[str, dict] = {}
        for st in set(r.get("strategy_type", "") for r in records):
            st_trades = [r for r in records if r.get("strategy_type") == st]
            st_pnls = [r["pnl"] for r in st_trades]
            wins = [p for p in st_pnls if p > 0]
            strat_breakdown[st] = {
                "count": len(st_trades),
                "total_pnl": sum(st_pnls),
                "win_rate": len(wins) / len(st_pnls) if st_pnls else 0,
                "avg_pnl": float(np.mean(st_pnls)) if st_pnls else 0,
            }

        # Compute metrics
        trade_dicts = trades_df.to_dict("records") if len(trades_df) > 0 else []
        metrics = BacktestMetrics.compute_all(eq, trade_dicts, initial_capital)

        return OptionsBacktestResult(
            equity_curve=eq,
            options_trades=trades_df,
            greeks_pnl_attribution=greeks_attr,
            strategy_breakdown=strat_breakdown,
            metrics=metrics,
            initial_capital=initial_capital,
        )
