"""
IBKR Broker Client — Unit Tests
=================================

Fully mocked tests for IBKRBrokerClient and IBKRIVDataManager.
No live IB Gateway connection required.
"""

import sys
import types
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Provide a lightweight stub of the ``ib_insync`` package so the module can be
# imported without the real (pip-installed) library.  This is ONLY a fallback;
# when ib_insync IS installed we use it normally.
# ---------------------------------------------------------------------------
_REAL_IB_INSYNC = True
try:
    import ib_insync  # noqa: F401
except ImportError:
    _REAL_IB_INSYNC = False
    _ib_mod = types.ModuleType("ib_insync")

    class _FakeIB:
        def __init__(self, *a, **kw): ...
        def connect(self, *a, **kw): ...
        def disconnect(self, *a, **kw): ...
        def isConnected(self): return True
        def sleep(self, t): ...
        def accountSummary(self, account=None): return []
        def positions(self, account=None): return []
        def openTrades(self): return []
        def placeOrder(self, *a, **kw): ...
        def cancelOrder(self, *a, **kw): ...
        def qualifyContracts(self, *a, **kw): ...
        def reqSecDefOptParams(self, *a, **kw): return []
        def reqMktData(self, *a, **kw): return MagicMock()
        def cancelMktData(self, *a, **kw): ...
        def reqHistoricalData(self, *a, **kw): return []
        @property
        def client(self):
            m = MagicMock()
            m.serverVersion.return_value = 163
            return m

    _ib_mod.IB = _FakeIB
    for _name in ("Contract", "Stock", "Index", "Option",
                   "MarketOrder", "LimitOrder", "util"):
        setattr(_ib_mod, _name, MagicMock)
    sys.modules["ib_insync"] = _ib_mod

from src.brokers.base import AccountInfo, Bar, OptionContract, Order, Position  # noqa: E402
from src.brokers.ibkr_client import IBKRBrokerClient  # noqa: E402


class TestIBKRConnect(unittest.TestCase):
    """Connection / disconnection."""

    @patch("src.brokers.ibkr_client.IB")
    def test_connect_success(self, MockIB):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        client_mock = MagicMock()
        client_mock.serverVersion.return_value = 163
        type(ib).client = PropertyMock(return_value=client_mock)

        client = IBKRBrokerClient(host="127.0.0.1", port=4002, account="U22452226")
        client.ib = ib
        client.connect(max_retries=1)

        ib.connect.assert_called_once()

    @patch("src.brokers.ibkr_client.IB")
    def test_connect_retries_then_fails(self, MockIB):
        ib = MockIB.return_value
        ib.connect.side_effect = ConnectionRefusedError("refused")

        client = IBKRBrokerClient()
        client.ib = ib

        with self.assertRaises(ConnectionError):
            client.connect(max_retries=2)

        self.assertEqual(ib.connect.call_count, 2)

    @patch("src.brokers.ibkr_client.IB")
    def test_disconnect(self, MockIB):
        ib = MockIB.return_value
        ib.isConnected.return_value = True

        client = IBKRBrokerClient()
        client.ib = ib
        client.disconnect()

        ib.disconnect.assert_called_once()

    @patch("src.brokers.ibkr_client.IB")
    def test_reconnect_loop_fires_on_disconnection(self, MockIB):
        """Ensure the watchdog thread attempts reconnection."""
        client = IBKRBrokerClient()
        ib = MagicMock()
        ib.isConnected.return_value = False
        ib.connect.return_value = None
        client.ib = ib

        # Run one iteration manually
        client._stop_reconnect = MagicMock()
        client._stop_reconnect.is_set.side_effect = [False, False, True]
        client._stop_reconnect.wait.return_value = None
        client._reconnect_loop()

        ib.connect.assert_called()


class TestIBKRAccount(unittest.TestCase):
    """Account and position queries."""

    def _make_client(self):
        client = IBKRBrokerClient(account="U22452226")
        client.ib = MagicMock()
        return client

    def test_get_account(self):
        client = self._make_client()
        tag_cash = MagicMock(tag="TotalCashValue", value="50000")
        tag_bp = MagicMock(tag="BuyingPower", value="100000")
        tag_nlv = MagicMock(tag="NetLiquidation", value="75000")
        client.ib.accountSummary.return_value = [tag_cash, tag_bp, tag_nlv]

        acct = client.get_account()
        self.assertIsInstance(acct, AccountInfo)
        self.assertEqual(acct.cash, 50_000)
        self.assertEqual(acct.buying_power, 100_000)
        self.assertEqual(acct.portfolio_value, 75_000)

    def test_get_positions(self):
        client = self._make_client()
        pos_mock = MagicMock()
        pos_mock.contract.symbol = "AAPL"
        pos_mock.position = 100
        pos_mock.avgCost = 175.50
        client.ib.positions.return_value = [pos_mock]

        positions = client.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertIsInstance(positions[0], Position)
        self.assertEqual(positions[0].symbol, "AAPL")
        self.assertEqual(positions[0].qty, 100)
        self.assertEqual(positions[0].side, "long")


class TestIBKROrders(unittest.TestCase):
    """Order placement and cancellation."""

    def _make_client(self):
        client = IBKRBrokerClient(account="U22452226")
        client.ib = MagicMock()
        return client

    def test_place_market_order(self):
        client = self._make_client()
        trade_mock = MagicMock()
        trade_mock.order.orderId = 42
        trade_mock.orderStatus.status = "Submitted"
        trade_mock.orderStatus.filled = 0
        trade_mock.orderStatus.avgFillPrice = 0
        client.ib.placeOrder.return_value = trade_mock

        order = client.place_order("AAPL", 10, "buy", "market")
        self.assertIsInstance(order, Order)
        self.assertEqual(order.order_id, "42")
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, "buy")
        client.ib.placeOrder.assert_called_once()

    def test_place_limit_order(self):
        client = self._make_client()
        trade_mock = MagicMock()
        trade_mock.order.orderId = 43
        trade_mock.orderStatus.status = "PreSubmitted"
        trade_mock.orderStatus.filled = 0
        trade_mock.orderStatus.avgFillPrice = 0
        client.ib.placeOrder.return_value = trade_mock

        order = client.place_order("AAPL", 5, "sell", "limit", limit_price=180.0)
        self.assertIsInstance(order, Order)
        self.assertEqual(order.order_type, "limit")

    def test_cancel_order(self):
        client = self._make_client()
        trade_mock = MagicMock()
        trade_mock.order.orderId = 42
        client.ib.openTrades.return_value = [trade_mock]

        result = client.cancel_order("42")
        self.assertTrue(result)
        client.ib.cancelOrder.assert_called_once()

    def test_cancel_order_not_found(self):
        client = self._make_client()
        client.ib.openTrades.return_value = []

        result = client.cancel_order("999")
        self.assertFalse(result)


class TestIBKROptionChain(unittest.TestCase):
    """Option chain with LIVE Greeks."""

    def _make_client(self):
        client = IBKRBrokerClient(account="U22452226")
        client.ib = MagicMock()
        return client

    def test_get_option_chain_returns_option_contracts(self):
        client = self._make_client()

        # Mock chain definition
        chain_def = MagicMock()
        chain_def.exchange = "SMART"
        chain_def.expirations = ["20260320"]
        chain_def.strikes = [400.0, 405.0]
        client.ib.reqSecDefOptParams.return_value = [chain_def]

        # Mock qualifyContracts to be a no-op
        client.ib.qualifyContracts.return_value = []

        # Mock reqMktData
        greeks_mock = MagicMock()
        greeks_mock.impliedVol = 0.22
        greeks_mock.delta = 0.55
        greeks_mock.gamma = 0.03
        greeks_mock.theta = -0.05
        greeks_mock.vega = 0.15

        ticker_mock = MagicMock()
        ticker_mock.modelGreeks = greeks_mock
        ticker_mock.lastGreeks = None
        ticker_mock.bid = 5.20
        ticker_mock.ask = 5.40
        ticker_mock.last = 5.30
        ticker_mock.volume = 1500
        client.ib.reqMktData.return_value = ticker_mock

        chain = client.get_option_chain("SPY", "20260320")

        # 2 strikes × 2 rights = 4 contracts
        self.assertEqual(len(chain), 4)
        for oc in chain:
            self.assertIsInstance(oc, OptionContract)
            self.assertEqual(oc.implied_volatility, 0.22)
            self.assertEqual(oc.delta, 0.55)
            self.assertEqual(oc.gamma, 0.03)
            self.assertEqual(oc.theta, -0.05)
            self.assertEqual(oc.vega, 0.15)
            self.assertGreater(oc.bid, 0)
            self.assertGreater(oc.ask, 0)

    def test_get_option_chain_no_chains(self):
        client = self._make_client()
        client.ib.reqSecDefOptParams.return_value = []

        chain = client.get_option_chain("XYZ", "20260320")
        self.assertEqual(chain, [])

    def test_get_option_chain_wrong_expiry(self):
        client = self._make_client()
        chain_def = MagicMock()
        chain_def.exchange = "SMART"
        chain_def.expirations = ["20260320"]
        chain_def.strikes = [100.0]
        client.ib.reqSecDefOptParams.return_value = [chain_def]

        chain = client.get_option_chain("SPY", "20271231")
        self.assertEqual(chain, [])


class TestIBKRVix(unittest.TestCase):
    """VIX quote."""

    def test_get_vix_returns_float(self):
        client = IBKRBrokerClient()
        client.ib = MagicMock()

        ticker_mock = MagicMock()
        ticker_mock.last = 18.5
        ticker_mock.close = 18.0
        client.ib.reqMktData.return_value = ticker_mock

        vix = client.get_vix()
        self.assertIsInstance(vix, float)
        self.assertAlmostEqual(vix, 18.5)


class TestIBKRIVDataManager(unittest.TestCase):
    """IBKRIVDataManager overrides."""

    def _make_manager(self):
        from src.options.iv_data_manager import IBKRIVDataManager
        mock_client = MagicMock()
        manager = IBKRIVDataManager(ibkr_client=mock_client, data_dir="/tmp/test_iv_cache")
        return manager

    def test_is_synthetic_always_false(self):
        mgr = self._make_manager()
        self.assertFalse(mgr.is_synthetic("SPY"))
        self.assertFalse(mgr.is_synthetic(""))
        self.assertFalse(mgr.is_synthetic())

    def test_data_quality_score_always_one(self):
        mgr = self._make_manager()
        self.assertEqual(mgr.data_quality_score("SPY"), 1.0)
        self.assertEqual(mgr.data_quality_score(""), 1.0)
        self.assertEqual(mgr.data_quality_score(), 1.0)

    def test_get_option_chain_with_iv_delegates(self):
        from src.options.iv_data_manager import IBKRIVDataManager
        mock_client = MagicMock()
        mock_client.get_option_chain.return_value = [
            OptionContract(
                symbol="AAPL", expiry="20260320", strike=200.0, right="C",
                bid=3.0, ask=3.2, implied_volatility=0.30,
                delta=0.50, gamma=0.02, theta=-0.04, vega=0.12,
            )
        ]
        mgr = IBKRIVDataManager(ibkr_client=mock_client, data_dir="/tmp/test_iv_cache")

        chain = mgr.get_option_chain_with_iv("AAPL", "20260320")
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].implied_volatility, 0.30)
        mock_client.get_option_chain.assert_called_once_with("AAPL", "20260320")


class TestIBKRBars(unittest.TestCase):
    """Historical bar data."""

    def test_get_bars_returns_bars(self):
        client = IBKRBrokerClient()
        client.ib = MagicMock()

        bar_mock = MagicMock()
        bar_mock.date = datetime(2026, 2, 20)
        bar_mock.open = 500.0
        bar_mock.high = 505.0
        bar_mock.low = 498.0
        bar_mock.close = 503.0
        bar_mock.volume = 100_000
        client.ib.reqHistoricalData.return_value = [bar_mock]

        bars = client.get_bars("SPY", "1D", 1)
        self.assertEqual(len(bars), 1)
        self.assertIsInstance(bars[0], Bar)
        self.assertEqual(bars[0].close, 503.0)


if __name__ == "__main__":
    unittest.main()
