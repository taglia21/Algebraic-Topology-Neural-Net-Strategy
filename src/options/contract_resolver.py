"""
Option Contract Discovery and Resolution
==========================================

Resolves trading signals to real, tradable OCC option contracts via Alpaca API.

This module bridges the gap between abstract signals (e.g., "sell SPY put, 30 DTE")
and real OCC symbols (e.g., "SPY250307P00580000") with live bid/ask pricing.

Three resolution modes:
- Single-leg: One call or put contract
- Spread: Two-leg credit/debit spread
- Iron condor: Four-leg strategy

All resolution functions return None on failure (never raise).
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

from .config import RISK_CONFIG


logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ResolvedContract:
    """A single resolved option contract with live pricing."""
    occ_symbol: str          # e.g., "SPY250307P00580000"
    underlying: str          # e.g., "SPY"
    strike: float            # e.g., 580.0
    expiration: date         # e.g., date(2025, 3, 7)
    option_type: str         # "call" or "put"
    bid: float               # Live bid price
    ask: float               # Live ask price
    mid_price: float         # (bid + ask) / 2
    open_interest: int       # Open interest (0 if unavailable)


@dataclass
class ResolvedSpread:
    """A two-leg spread with pricing."""
    long_leg: ResolvedContract
    short_leg: ResolvedContract
    net_credit: float        # Credit received (positive) or debit paid (negative)
    max_loss: float          # Maximum loss per spread
    max_profit: float        # Maximum profit per spread


@dataclass
class ResolvedIronCondor:
    """A four-leg iron condor with pricing."""
    put_spread: ResolvedSpread
    call_spread: ResolvedSpread
    total_credit: float      # Total premium collected
    max_loss: float          # Maximum loss


# ============================================================================
# WING WIDTH CONFIGURATION
# ============================================================================

# Wing width in strike points per underlying
WING_WIDTHS: Dict[str, float] = {
    "SPY": 5.0,
    "QQQ": 5.0,
    "IWM": 5.0,
}
DEFAULT_WING_WIDTH = 2.5


def _get_wing_width(symbol: str) -> float:
    """Get the wing width for a given underlying symbol."""
    return WING_WIDTHS.get(symbol, DEFAULT_WING_WIDTH)


# ============================================================================
# CONTRACT RESOLVER
# ============================================================================

class OptionContractResolver:
    """
    Resolve abstract trading signals to real OCC option contracts.

    Uses Alpaca's get_option_contracts() to discover tradable contracts,
    then fetches live bid/ask quotes to derive limit prices.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: OptionHistoricalDataClient,
    ) -> None:
        """
        Initialize resolver with Alpaca API clients.

        Args:
            trading_client: Alpaca TradingClient (for contract discovery).
            data_client: Alpaca OptionHistoricalDataClient (for quotes).
        """
        self.trading_client = trading_client
        self.data_client = data_client
        self.logger = logging.getLogger(__name__)

        # Cache for current underlying prices (symbol -> price)
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._price_cache_ttl = timedelta(seconds=30)

    # ------------------------------------------------------------------ #
    # PUBLIC: Single Leg
    # ------------------------------------------------------------------ #

    async def resolve_single_leg(
        self,
        symbol: str,
        option_type: str,
        target_dte: int,
        target_delta: float = 0.30,
        target_strike: Optional[float] = None,
    ) -> Optional[ResolvedContract]:
        """
        Resolve a single-leg option to a real OCC contract.

        Steps:
        1. Query Alpaca for active contracts matching symbol, type, DTE window.
        2. Filter by open interest > 10 and tradability.
        3. Select strike closest to target (delta-approximated or explicit).
        4. Fetch live bid/ask quote.
        5. Return ResolvedContract or None.

        Args:
            symbol: Underlying ticker (e.g., "SPY").
            option_type: "call" or "put".
            target_dte: Desired days to expiration.
            target_delta: Target delta (used to approximate strike). Default 0.30.
            target_strike: If provided, pick the closest strike to this value.

        Returns:
            ResolvedContract if found, None otherwise.
        """
        self.logger.info(
            f"Resolving single leg: {symbol} {option_type} "
            f"~{target_dte}DTE delta={target_delta:.2f}"
        )

        try:
            contracts = await self._fetch_contracts(
                symbol=symbol,
                option_type=option_type,
                target_dte=target_dte,
            )

            if not contracts:
                self.logger.warning(
                    f"No contracts found for {symbol} {option_type} ~{target_dte}DTE"
                )
                return None

            # Determine the target strike price
            if target_strike is None:
                current_price = await self._get_underlying_price(symbol)
                if current_price is None:
                    self.logger.warning(f"Cannot get price for {symbol}, cannot resolve")
                    return None
                target_strike = self._estimate_strike(
                    current_price, option_type, target_delta
                )

            # Pick best contract
            best = self._select_best_contract(contracts, target_strike)
            if best is None:
                self.logger.warning(f"No suitable contract after filtering for {symbol}")
                return None

            # Fetch live quote
            resolved = await self._hydrate_contract(best, symbol, option_type)
            if resolved is None:
                return None

            self.logger.info(
                f"Resolved {symbol} {option_type}: {resolved.occ_symbol} "
                f"strike={resolved.strike} exp={resolved.expiration} "
                f"bid={resolved.bid:.2f} ask={resolved.ask:.2f} mid={resolved.mid_price:.2f}"
            )
            return resolved

        except Exception as e:
            self.logger.warning(
                f"Failed to resolve single leg {symbol} {option_type}: {e}",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------ #
    # PUBLIC: Spread
    # ------------------------------------------------------------------ #

    async def resolve_spread(
        self,
        symbol: str,
        spread_type: str,
        target_dte: int,
        target_delta: float = 0.30,
    ) -> Optional[ResolvedSpread]:
        """
        Resolve a two-leg spread to real OCC contracts.

        For credit put spread (bull put):
          - Short leg: near-ATM put (higher strike, closer to current price)
          - Long leg:  OTM protective put (lower strike, further from price)

        For credit call spread (bear call):
          - Short leg: near-ATM call (lower strike, closer to current price)
          - Long leg:  OTM protective call (higher strike, further from price)

        Args:
            symbol: Underlying ticker.
            spread_type: "credit_spread", "put_spread", "call_spread", or "debit_spread".
            target_dte: Desired days to expiration.
            target_delta: Target delta for the short leg.

        Returns:
            ResolvedSpread if both legs resolve, None otherwise.
        """
        self.logger.info(
            f"Resolving spread: {symbol} {spread_type} ~{target_dte}DTE"
        )

        try:
            current_price = await self._get_underlying_price(symbol)
            if current_price is None:
                self.logger.warning(f"Cannot get price for {symbol}")
                return None

            wing = _get_wing_width(symbol)

            # Determine option type and strikes based on spread type
            if spread_type in ("put_spread", "credit_spread"):
                option_type = "put"
                # Short put closer to ATM, long put further OTM
                short_strike_target = self._estimate_strike(
                    current_price, "put", target_delta
                )
                long_strike_target = short_strike_target - wing
            elif spread_type == "call_spread":
                option_type = "call"
                short_strike_target = self._estimate_strike(
                    current_price, "call", target_delta
                )
                long_strike_target = short_strike_target + wing
            else:
                # Default to put credit spread
                option_type = "put"
                short_strike_target = self._estimate_strike(
                    current_price, "put", target_delta
                )
                long_strike_target = short_strike_target - wing

            # Resolve both legs
            short_leg = await self.resolve_single_leg(
                symbol=symbol,
                option_type=option_type,
                target_dte=target_dte,
                target_strike=short_strike_target,
            )
            if short_leg is None:
                self.logger.warning(f"Failed to resolve short leg for {symbol} spread")
                return None

            long_leg = await self.resolve_single_leg(
                symbol=symbol,
                option_type=option_type,
                target_dte=target_dte,
                target_strike=long_strike_target,
            )
            if long_leg is None:
                self.logger.warning(f"Failed to resolve long leg for {symbol} spread")
                return None

            # Ensure legs are different contracts
            if short_leg.occ_symbol == long_leg.occ_symbol:
                self.logger.warning(f"Short and long legs resolved to same contract for {symbol}")
                return None

            # Calculate spread economics
            # Credit spread: sell the expensive leg, buy the cheap protection
            net_credit = short_leg.mid_price - long_leg.mid_price
            strike_width = abs(short_leg.strike - long_leg.strike)
            max_loss = (strike_width - net_credit) * 100  # per contract
            max_profit = net_credit * 100

            # Sanity: net_credit should be positive for credit spread
            if net_credit <= 0:
                self.logger.warning(
                    f"Spread {symbol} has non-positive credit: "
                    f"${net_credit:.2f} (short mid={short_leg.mid_price:.2f}, "
                    f"long mid={long_leg.mid_price:.2f})"
                )
                # Still return it — the engine can decide to skip

            spread = ResolvedSpread(
                long_leg=long_leg,
                short_leg=short_leg,
                net_credit=round(net_credit, 2),
                max_loss=round(max_loss, 2),
                max_profit=round(max_profit, 2),
            )

            self.logger.info(
                f"Resolved spread {symbol}: short={short_leg.occ_symbol} "
                f"long={long_leg.occ_symbol} credit=${net_credit:.2f}"
            )
            return spread

        except Exception as e:
            self.logger.warning(
                f"Failed to resolve spread {symbol} {spread_type}: {e}",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------ #
    # PUBLIC: Iron Condor
    # ------------------------------------------------------------------ #

    async def resolve_iron_condor(
        self,
        symbol: str,
        target_dte: int,
        target_delta: float = 0.20,
    ) -> Optional[ResolvedIronCondor]:
        """
        Resolve a four-leg iron condor to real OCC contracts.

        Structure:
        - Put spread:  sell put (higher strike), buy put (lower strike)
        - Call spread: sell call (lower strike), buy call (higher strike)

        Wing width is determined per-symbol (5 pts for SPY/QQQ, 2.5 default).

        Args:
            symbol: Underlying ticker.
            target_dte: Desired days to expiration.
            target_delta: Target delta for the short strikes.

        Returns:
            ResolvedIronCondor if all four legs resolve, None otherwise.
        """
        self.logger.info(
            f"Resolving iron condor: {symbol} ~{target_dte}DTE delta={target_delta:.2f}"
        )

        try:
            # Resolve the put credit spread (lower side)
            put_spread = await self.resolve_spread(
                symbol=symbol,
                spread_type="put_spread",
                target_dte=target_dte,
                target_delta=target_delta,
            )
            if put_spread is None:
                self.logger.warning(f"Failed to resolve put spread for {symbol} IC")
                return None

            # Resolve the call credit spread (upper side)
            call_spread = await self.resolve_spread(
                symbol=symbol,
                spread_type="call_spread",
                target_dte=target_dte,
                target_delta=target_delta,
            )
            if call_spread is None:
                self.logger.warning(f"Failed to resolve call spread for {symbol} IC")
                return None

            total_credit = put_spread.net_credit + call_spread.net_credit
            # Max loss is the wider spread width minus total credit
            max_loss = max(put_spread.max_loss, call_spread.max_loss)

            ic = ResolvedIronCondor(
                put_spread=put_spread,
                call_spread=call_spread,
                total_credit=round(total_credit, 2),
                max_loss=round(max_loss, 2),
            )

            self.logger.info(
                f"Resolved iron condor {symbol}: "
                f"put_spread credit=${put_spread.net_credit:.2f} "
                f"call_spread credit=${call_spread.net_credit:.2f} "
                f"total=${total_credit:.2f}"
            )
            return ic

        except Exception as e:
            self.logger.warning(
                f"Failed to resolve iron condor {symbol}: {e}",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------ #
    # PRIVATE: Contract Fetching
    # ------------------------------------------------------------------ #

    async def _fetch_contracts(
        self,
        symbol: str,
        option_type: str,
        target_dte: int,
    ) -> List:
        """
        Fetch option contracts from Alpaca matching criteria.

        Uses a DTE window of [target_dte - 7, target_dte + 14] to allow
        flexibility in expiration selection.

        Args:
            symbol: Underlying ticker.
            option_type: "call" or "put".
            target_dte: Target DTE for the search window.

        Returns:
            List of OptionContract objects (may be empty).
        """
        today = date.today()
        # Clamp min DTE to config minimum
        min_dte = max(RISK_CONFIG.get("min_dte", 7), target_dte - 7)
        max_dte = target_dte + 14

        exp_gte = today + timedelta(days=min_dte)
        exp_lte = today + timedelta(days=max_dte)

        contract_type = (
            ContractType.CALL if option_type.lower() == "call" else ContractType.PUT
        )

        request = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=exp_gte.isoformat(),
            expiration_date_lte=exp_lte.isoformat(),
            type=contract_type,
            limit=100,
        )

        self.logger.debug(
            f"Fetching contracts: {symbol} {option_type} "
            f"exp=[{exp_gte}, {exp_lte}]"
        )

        # Alpaca SDK call (synchronous)
        response = await asyncio.to_thread(
            self.trading_client.get_option_contracts, request
        )

        # Handle both response types
        if hasattr(response, "option_contracts") and response.option_contracts:
            contracts = response.option_contracts
        elif isinstance(response, dict):
            contracts = response.get("option_contracts", [])
        else:
            contracts = []

        self.logger.debug(f"Fetched {len(contracts)} raw contracts for {symbol} {option_type}")

        # Filter: tradable and open_interest > 10
        filtered = []
        for c in contracts:
            if not getattr(c, "tradable", True):
                continue

            oi_raw = getattr(c, "open_interest", None)
            oi = int(oi_raw) if oi_raw is not None else 0
            if oi < 10:
                continue

            filtered.append(c)

        self.logger.debug(
            f"After filtering: {len(filtered)} contracts "
            f"(dropped {len(contracts) - len(filtered)} illiquid/non-tradable)"
        )
        return filtered

    # ------------------------------------------------------------------ #
    # PRIVATE: Strike Selection
    # ------------------------------------------------------------------ #

    def _estimate_strike(
        self,
        current_price: float,
        option_type: str,
        target_delta: float,
    ) -> float:
        """
        Approximate a target strike from delta using a simple offset.

        Assumption: A delta of 0.30 corresponds to roughly ±8% OTM for 30-DTE.
        This is a heuristic; real delta would require a full BSM calculation.

        Args:
            current_price: Current underlying price.
            option_type: "call" or "put".
            target_delta: Target delta (0.0 – 1.0).

        Returns:
            Estimated strike price.
        """
        # Map delta to approximate OTM percentage
        # delta ~0.50 -> ATM, delta ~0.30 -> ~5-8% OTM, delta ~0.16 -> ~10-15% OTM
        otm_pct = (0.50 - target_delta) * 0.30  # Rough mapping

        if option_type.lower() == "put":
            return round(current_price * (1 - otm_pct), 2)
        else:  # call
            return round(current_price * (1 + otm_pct), 2)

    def _select_best_contract(
        self,
        contracts: List,
        target_strike: float,
    ) -> Optional[object]:
        """
        Select the contract with strike closest to target.

        Args:
            contracts: Filtered list of OptionContract objects.
            target_strike: Desired strike price.

        Returns:
            Best OptionContract or None.
        """
        if not contracts:
            return None

        best = min(
            contracts,
            key=lambda c: abs(float(getattr(c, "strike_price", 0)) - target_strike),
        )
        return best

    # ------------------------------------------------------------------ #
    # PRIVATE: Quote Fetching & Hydration
    # ------------------------------------------------------------------ #

    async def _hydrate_contract(
        self,
        contract: object,
        underlying: str,
        option_type: str,
    ) -> Optional[ResolvedContract]:
        """
        Fetch a live quote for a contract and return a ResolvedContract.

        Args:
            contract: Alpaca OptionContract object.
            underlying: Underlying ticker.
            option_type: "call" or "put".

        Returns:
            ResolvedContract or None if quote unavailable.
        """
        occ_symbol = getattr(contract, "symbol", None)
        if not occ_symbol:
            self.logger.warning("Contract missing symbol attribute")
            return None

        strike = float(getattr(contract, "strike_price", 0))
        expiration = getattr(contract, "expiration_date", date.today())
        oi_raw = getattr(contract, "open_interest", None)
        oi = int(oi_raw) if oi_raw is not None else 0

        # Fetch live bid/ask
        bid, ask = await self._get_option_quote(occ_symbol)

        # Fallback: if no quote, use close_price as estimate
        if bid == 0.0 and ask == 0.0:
            close_raw = getattr(contract, "close_price", None)
            if close_raw is not None:
                close = float(close_raw)
                # Synthesize a tight spread around close
                bid = round(close * 0.95, 2)
                ask = round(close * 1.05, 2)
                self.logger.debug(
                    f"No live quote for {occ_symbol}, using close-based estimate: "
                    f"bid={bid:.2f} ask={ask:.2f}"
                )
            else:
                self.logger.warning(f"No quote or close price for {occ_symbol}")
                return None

        mid_price = round((bid + ask) / 2, 2)

        # Validate: reject nonsensical quotes
        if mid_price <= 0:
            self.logger.warning(f"Invalid mid price {mid_price} for {occ_symbol}")
            return None

        # Check bid-ask spread width
        spread_pct = (ask - bid) / mid_price if mid_price > 0 else 999
        max_spread = RISK_CONFIG.get("max_bid_ask_spread_pct", 0.15)
        if spread_pct > max_spread:
            self.logger.warning(
                f"Wide bid-ask spread for {occ_symbol}: "
                f"{spread_pct:.1%} > {max_spread:.0%} limit "
                f"(bid={bid:.2f} ask={ask:.2f})"
            )
            # Still return it — let the engine decide whether to skip

        return ResolvedContract(
            occ_symbol=occ_symbol,
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            bid=bid,
            ask=ask,
            mid_price=mid_price,
            open_interest=oi,
        )

    async def _get_option_quote(self, occ_symbol: str) -> Tuple[float, float]:
        """
        Fetch latest bid/ask for an OCC option symbol.

        Args:
            occ_symbol: OCC-format symbol.

        Returns:
            (bid, ask) tuple. (0.0, 0.0) if unavailable.
        """
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol)
            quotes = await asyncio.to_thread(
                self.data_client.get_option_latest_quote, request
            )

            if occ_symbol in quotes:
                q = quotes[occ_symbol]
                bid = float(q.bid_price) if q.bid_price else 0.0
                ask = float(q.ask_price) if q.ask_price else 0.0
                return bid, ask

            return 0.0, 0.0

        except Exception as e:
            self.logger.debug(f"Quote fetch failed for {occ_symbol}: {e}")
            return 0.0, 0.0

    # ------------------------------------------------------------------ #
    # PRIVATE: Underlying Price
    # ------------------------------------------------------------------ #

    async def _get_underlying_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price of the underlying via Alpaca.

        Uses a short-lived cache (30s TTL) to avoid hammering the API.

        Args:
            symbol: Underlying ticker.

        Returns:
            Current price or None.
        """
        # Check cache
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if datetime.now() - ts < self._price_cache_ttl:
                return price

        try:
            # Use the trading client to get latest trade/snapshot
            # For equities, get_latest_bar or get_snapshot would work,
            # but we use the trading client's asset info + a stock data call
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestBarRequest

            stock_client = StockHistoricalDataClient(
                api_key=self.trading_client._api_key
                if hasattr(self.trading_client, "_api_key")
                else None,
                secret_key=self.trading_client._secret_key
                if hasattr(self.trading_client, "_secret_key")
                else None,
            )
            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = await asyncio.to_thread(stock_client.get_stock_latest_bar, request)

            if symbol in bars:
                price = float(bars[symbol].close)
                self._price_cache[symbol] = (price, datetime.now())
                self.logger.debug(f"Underlying price {symbol}: ${price:.2f}")
                return price

            return None

        except Exception as e:
            self.logger.warning(f"Failed to get price for {symbol}: {e}")
            # Fallback: try to infer from option strikes
            return None
