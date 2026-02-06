"""
Unit Tests for OptionContractResolver
======================================

Tests contract resolution with mocked Alpaca API responses.
Validates OCC symbol format, spread leg relationships, iron condor structure,
graceful failure on empty results, and mid-price calculation.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.options.contract_resolver import (
    DEFAULT_WING_WIDTH,
    OptionContractResolver,
    ResolvedContract,
    ResolvedIronCondor,
    ResolvedSpread,
    _get_wing_width,
)

# OCC symbol regex: 1-6 uppercase letters, 6 digits (YYMMDD), C or P, 8 digits
OCC_PATTERN = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")


# ============================================================================
# HELPERS — Mock Alpaca objects
# ============================================================================

@dataclass
class _MockOptionContract:
    """Mimics alpaca.trading.models.OptionContract."""
    symbol: str
    strike_price: float
    expiration_date: date
    type: str            # "call" or "put"
    tradable: bool = True
    open_interest: Optional[str] = "500"
    close_price: Optional[str] = "2.50"
    underlying_symbol: str = "SPY"
    root_symbol: str = "SPY"
    name: str = ""
    status: str = "active"
    id: str = "mock-id"
    style: str = "american"
    size: str = "100"
    underlying_asset_id: str = "00000000-0000-0000-0000-000000000000"
    open_interest_date: Optional[date] = None
    close_price_date: Optional[date] = None


@dataclass
class _MockOptionContractsResponse:
    """Mimics alpaca.trading.models.OptionContractsResponse."""
    option_contracts: List[_MockOptionContract]
    next_page_token: Optional[str] = None


@dataclass
class _MockQuote:
    """Mimics alpaca option quote."""
    bid_price: float
    ask_price: float
    bid_size: int = 100
    ask_size: int = 100


@dataclass
class _MockBar:
    """Mimics alpaca stock bar."""
    close: float


def _make_contracts(
    symbol: str = "SPY",
    option_type: str = "put",
    base_strike: float = 580.0,
    count: int = 5,
    strike_step: float = 5.0,
    exp: Optional[date] = None,
) -> List[_MockOptionContract]:
    """Generate a list of mock option contracts."""
    if exp is None:
        exp = date.today() + timedelta(days=30)
    type_char = "P" if option_type == "put" else "C"
    contracts = []
    for i in range(count):
        strike = base_strike + i * strike_step
        # Build a valid OCC symbol
        occ = f"{symbol}{exp.strftime('%y%m%d')}{type_char}{int(strike * 1000):08d}"
        contracts.append(
            _MockOptionContract(
                symbol=occ,
                strike_price=strike,
                expiration_date=exp,
                type=option_type,
                underlying_symbol=symbol,
                open_interest=str(100 + i * 50),
            )
        )
    return contracts


def _build_resolver(
    contracts_response=None,
    quote_map=None,
    bar_map=None,
) -> OptionContractResolver:
    """Build a resolver with fully mocked clients."""
    trading_client = MagicMock()
    data_client = MagicMock()

    # Mock get_option_contracts
    if contracts_response is not None:
        trading_client.get_option_contracts.return_value = contracts_response
    else:
        trading_client.get_option_contracts.return_value = _MockOptionContractsResponse(
            option_contracts=[]
        )

    # Mock get_option_latest_quote
    if quote_map is None:
        quote_map = {}
    data_client.get_option_latest_quote.return_value = quote_map

    # We need to set attributes for price fetching
    trading_client._api_key = "test_key"
    trading_client._secret_key = "test_secret"

    resolver = OptionContractResolver(trading_client, data_client)

    # Patch _get_underlying_price to avoid real API calls
    if bar_map:
        async def _mock_price(sym):
            return bar_map.get(sym)
        resolver._get_underlying_price = _mock_price
    else:
        async def _mock_price(sym):
            return 590.0  # Default SPY price
        resolver._get_underlying_price = _mock_price

    return resolver


# ============================================================================
# TEST: OCC Symbol Format
# ============================================================================

class TestOCCFormat:
    """Verify resolved contracts have valid OCC symbols."""

    def test_occ_pattern_matches_valid_symbols(self):
        assert OCC_PATTERN.match("SPY250307P00580000")
        assert OCC_PATTERN.match("AAPL260115C00200000")
        assert OCC_PATTERN.match("QQQ250221C00500000")

    def test_occ_pattern_rejects_invalid_symbols(self):
        assert not OCC_PATTERN.match("SPY_CALL_100")
        assert not OCC_PATTERN.match("SPY250307X00580000")  # Invalid type
        assert not OCC_PATTERN.match("")

    def test_resolve_single_leg_returns_valid_occ(self):
        exp = date.today() + timedelta(days=30)
        contracts = _make_contracts("SPY", "put", 580.0, count=5, exp=exp)
        response = _MockOptionContractsResponse(option_contracts=contracts)

        # Build quote map for the contracts
        quote_map = {}
        for c in contracts:
            quote_map[c.symbol] = _MockQuote(bid_price=2.10, ask_price=2.30)

        resolver = _build_resolver(
            contracts_response=response,
            quote_map=quote_map,
            bar_map={"SPY": 590.0},
        )

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_single_leg("SPY", "put", target_dte=30)
        )

        assert result is not None
        assert OCC_PATTERN.match(result.occ_symbol), f"Invalid OCC: {result.occ_symbol}"
        assert result.underlying == "SPY"
        assert result.option_type == "put"


# ============================================================================
# TEST: Spread Leg Relationships
# ============================================================================

class TestSpreadResolution:
    """Verify spread legs have correct structure."""

    def test_put_credit_spread_structure(self):
        exp = date.today() + timedelta(days=30)
        # Contracts from 550 to 600 with $2.50 spacing for finer granularity
        contracts = _make_contracts("SPY", "put", 550.0, count=21, strike_step=2.5, exp=exp)
        response = _MockOptionContractsResponse(option_contracts=contracts)

        quote_map = {}
        for c in contracts:
            # Higher strike puts are worth more (closer to ATM)
            premium = max(0.20, (600 - c.strike_price) * 0.08 + 0.30)
            quote_map[c.symbol] = _MockQuote(
                bid_price=round(premium - 0.10, 2),
                ask_price=round(premium + 0.10, 2),
            )

        resolver = _build_resolver(
            contracts_response=response,
            quote_map=quote_map,
            bar_map={"SPY": 595.0},
        )

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_spread("SPY", "put_spread", target_dte=30)
        )

        assert result is not None
        # Short leg should be closer to ATM (higher strike for puts)
        assert result.short_leg.strike >= result.long_leg.strike, (
            f"Short strike {result.short_leg.strike} should be >= "
            f"long strike {result.long_leg.strike} for put credit spread"
        )
        # Both legs should have valid OCC symbols
        assert OCC_PATTERN.match(result.short_leg.occ_symbol)
        assert OCC_PATTERN.match(result.long_leg.occ_symbol)
        # Legs should be different
        assert result.short_leg.occ_symbol != result.long_leg.occ_symbol

    def test_spread_max_loss_calculation(self):
        spread = ResolvedSpread(
            long_leg=ResolvedContract(
                occ_symbol="SPY250307P00575000",
                underlying="SPY",
                strike=575.0,
                expiration=date.today(),
                option_type="put",
                bid=1.00,
                ask=1.20,
                mid_price=1.10,
                open_interest=200,
            ),
            short_leg=ResolvedContract(
                occ_symbol="SPY250307P00580000",
                underlying="SPY",
                strike=580.0,
                expiration=date.today(),
                option_type="put",
                bid=2.00,
                ask=2.20,
                mid_price=2.10,
                open_interest=300,
            ),
            net_credit=1.00,
            max_loss=400.0,    # ($5 width - $1 credit) * 100
            max_profit=100.0,  # $1 credit * 100
        )
        assert spread.max_loss == 400.0
        assert spread.max_profit == 100.0


# ============================================================================
# TEST: Iron Condor Structure
# ============================================================================

class TestIronCondorResolution:
    """Verify iron condor has proper 4-leg structure."""

    def test_iron_condor_has_four_legs(self):
        exp = date.today() + timedelta(days=30)

        # Put contracts — must extend low enough for delta=0.20 (targets ~536-541)
        put_contracts = _make_contracts("SPY", "put", 530.0, count=25, strike_step=2.5, exp=exp)
        # Call contracts — must extend high enough (targets ~648-653)
        call_contracts = _make_contracts("SPY", "call", 640.0, count=25, strike_step=2.5, exp=exp)
        all_contracts = put_contracts + call_contracts

        response = _MockOptionContractsResponse(option_contracts=all_contracts)

        quote_map = {}
        for c in all_contracts:
            quote_map[c.symbol] = _MockQuote(bid_price=1.50, ask_price=1.80)

        resolver = _build_resolver(
            contracts_response=response,
            quote_map=quote_map,
            bar_map={"SPY": 595.0},
        )

        # The resolver fetches contracts per type, so we need to return
        # different sets based on the request
        def _mock_get_contracts(request):
            contract_type = getattr(request, "type", None)
            if contract_type is not None:
                ct_val = contract_type.value if hasattr(contract_type, "value") else str(contract_type)
                if ct_val == "put":
                    return _MockOptionContractsResponse(option_contracts=put_contracts)
                else:
                    return _MockOptionContractsResponse(option_contracts=call_contracts)
            return response

        resolver.trading_client.get_option_contracts.side_effect = _mock_get_contracts

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_iron_condor("SPY", target_dte=30)
        )

        assert result is not None
        # Should have put spread and call spread
        assert result.put_spread is not None
        assert result.call_spread is not None
        # Put spread legs should both be puts
        assert result.put_spread.long_leg.option_type == "put"
        assert result.put_spread.short_leg.option_type == "put"
        # Call spread legs should both be calls
        assert result.call_spread.long_leg.option_type == "call"
        assert result.call_spread.short_leg.option_type == "call"
        # Total credit should be sum of both spreads
        expected_credit = result.put_spread.net_credit + result.call_spread.net_credit
        assert abs(result.total_credit - expected_credit) < 0.01


# ============================================================================
# TEST: Graceful Failure
# ============================================================================

class TestGracefulFailure:
    """Verify resolver returns None (never crashes) on empty/bad data."""

    def test_no_contracts_returns_none(self):
        resolver = _build_resolver(
            contracts_response=_MockOptionContractsResponse(option_contracts=[]),
        )

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_single_leg("SPY", "put", target_dte=30)
        )
        assert result is None

    def test_no_contracts_spread_returns_none(self):
        resolver = _build_resolver(
            contracts_response=_MockOptionContractsResponse(option_contracts=[]),
        )

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_spread("SPY", "put_spread", target_dte=30)
        )
        assert result is None

    def test_no_contracts_iron_condor_returns_none(self):
        resolver = _build_resolver(
            contracts_response=_MockOptionContractsResponse(option_contracts=[]),
        )

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_iron_condor("SPY", target_dte=30)
        )
        assert result is None

    def test_api_exception_returns_none(self):
        resolver = _build_resolver()
        resolver.trading_client.get_option_contracts.side_effect = Exception("API down")

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_single_leg("SPY", "put", target_dte=30)
        )
        assert result is None

    def test_no_price_returns_none(self):
        """If underlying price can't be fetched, resolve returns None."""
        exp = date.today() + timedelta(days=30)
        contracts = _make_contracts("XYZ", "put", 50.0, count=3, exp=exp)
        response = _MockOptionContractsResponse(option_contracts=contracts)

        resolver = _build_resolver(
            contracts_response=response,
            bar_map={},  # No price for any symbol
        )

        async def _no_price(sym):
            return None
        resolver._get_underlying_price = _no_price

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_single_leg("XYZ", "put", target_dte=30)
        )
        assert result is None

    def test_illiquid_contracts_filtered_out(self):
        """Contracts with open_interest < 10 should be filtered."""
        exp = date.today() + timedelta(days=30)
        contracts = [
            _MockOptionContract(
                symbol=f"SPY{exp.strftime('%y%m%d')}P00580000",
                strike_price=580.0,
                expiration_date=exp,
                type="put",
                open_interest="5",  # Below threshold
            ),
        ]
        response = _MockOptionContractsResponse(option_contracts=contracts)

        resolver = _build_resolver(contracts_response=response)

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_single_leg("SPY", "put", target_dte=30)
        )
        assert result is None


# ============================================================================
# TEST: Mid-Price Calculation
# ============================================================================

class TestMidPrice:
    """Verify mid_price = (bid + ask) / 2."""

    def test_mid_price_calculation(self):
        exp = date.today() + timedelta(days=30)
        contracts = _make_contracts("SPY", "call", 590.0, count=1, exp=exp)
        response = _MockOptionContractsResponse(option_contracts=contracts)

        bid, ask = 3.40, 3.80
        quote_map = {contracts[0].symbol: _MockQuote(bid_price=bid, ask_price=ask)}

        resolver = _build_resolver(
            contracts_response=response,
            quote_map=quote_map,
            bar_map={"SPY": 590.0},
        )

        result = asyncio.get_event_loop().run_until_complete(
            resolver.resolve_single_leg(
                "SPY", "call", target_dte=30, target_strike=590.0
            )
        )

        assert result is not None
        expected_mid = round((bid + ask) / 2, 2)
        assert result.mid_price == expected_mid, (
            f"Expected mid={expected_mid}, got {result.mid_price}"
        )
        assert result.bid == bid
        assert result.ask == ask


# ============================================================================
# TEST: Wing Width Configuration
# ============================================================================

class TestWingWidth:
    """Verify wing widths for different underlyings."""

    def test_spy_wing_width(self):
        assert _get_wing_width("SPY") == 5.0

    def test_qqq_wing_width(self):
        assert _get_wing_width("QQQ") == 5.0

    def test_default_wing_width(self):
        assert _get_wing_width("AAPL") == DEFAULT_WING_WIDTH
        assert _get_wing_width("TSLA") == DEFAULT_WING_WIDTH


# ============================================================================
# TEST: ResolvedContract Dataclass
# ============================================================================

class TestResolvedContract:
    """Verify dataclass fields."""

    def test_fields_present(self):
        rc = ResolvedContract(
            occ_symbol="SPY250307P00580000",
            underlying="SPY",
            strike=580.0,
            expiration=date(2025, 3, 7),
            option_type="put",
            bid=2.10,
            ask=2.30,
            mid_price=2.20,
            open_interest=500,
        )
        assert rc.occ_symbol == "SPY250307P00580000"
        assert rc.underlying == "SPY"
        assert rc.strike == 580.0
        assert rc.option_type == "put"
        assert rc.mid_price == 2.20
        assert rc.open_interest == 500
