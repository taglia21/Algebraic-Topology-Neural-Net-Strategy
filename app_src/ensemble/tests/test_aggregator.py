"""Tests for SignalAggregator."""

from __future__ import annotations

import pandas as pd
import pytest

from ensemble.signal_aggregator import SignalAggregator


class TestSignalAggregator:
    """Tests for SignalAggregator."""

    @pytest.fixture()
    def aggregator(self) -> SignalAggregator:
        return SignalAggregator()

    @pytest.fixture()
    def tda_signals(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"ticker": "AAPL", "direction": "LONG", "strength": 0.8, "timestamp": "2024-01-02"},
            {"ticker": "MSFT", "direction": "SHORT", "strength": 0.6, "timestamp": "2024-01-02"},
            {"ticker": "GOOG", "direction": "NEUTRAL", "strength": 0.0, "timestamp": "2024-01-02"},
        ])

    @pytest.fixture()
    def nn_signals(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"ticker": "AAPL", "direction": "LONG", "strength": 0.7, "timestamp": "2024-01-02"},
            {"ticker": "MSFT", "direction": "LONG", "strength": 0.5, "timestamp": "2024-01-02"},
            {"ticker": "TSLA", "direction": "SHORT", "strength": 0.9, "timestamp": "2024-01-02"},
        ])

    def test_agreement_bonus(self, aggregator, tda_signals, nn_signals):
        """When both agree, strength should be boosted by 20%."""
        result = aggregator.aggregate(tda_signals, nn_signals, 0.5, 0.5)
        aapl = result[result["ticker"] == "AAPL"].iloc[0]

        # Both say LONG: raw = 0.5*0.8 + 0.5*0.7 = 0.75, with bonus = 0.9
        assert bool(aapl["agreement"]) is True
        assert aapl["direction"] == "LONG"
        expected = min(1.0, 0.75 * 1.2)
        assert aapl["final_strength"] == pytest.approx(expected, abs=1e-4)

    def test_disagreement_penalty(self, aggregator, tda_signals, nn_signals):
        """When strategies disagree, strength should be reduced by 30%."""
        result = aggregator.aggregate(tda_signals, nn_signals, 0.5, 0.5)
        msft = result[result["ticker"] == "MSFT"].iloc[0]

        # TDA: SHORT 0.6, NN: LONG 0.5 → disagree
        assert bool(msft["agreement"]) is False
        # raw = 0.5*0.6 + 0.5*0.5 = 0.55, penalty → 0.55 * 0.7 = 0.385
        expected = 0.55 * 0.7
        assert msft["final_strength"] == pytest.approx(expected, abs=1e-4)

    def test_direction_resolution_stronger_wins(self, aggregator):
        """When disagreeing, stronger weighted signal determines direction."""
        tda = pd.DataFrame([
            {"ticker": "X", "direction": "SHORT", "strength": 0.9, "timestamp": "2024-01-02"},
        ])
        nn = pd.DataFrame([
            {"ticker": "X", "direction": "LONG", "strength": 0.3, "timestamp": "2024-01-02"},
        ])
        result = aggregator.aggregate(tda, nn, 0.7, 0.3)
        x_row = result[result["ticker"] == "X"].iloc[0]
        # TDA component = 0.7*0.9 = 0.63, NN component = 0.3*0.3 = 0.09
        assert x_row["direction"] == "SHORT"

    def test_equal_disagreement_goes_neutral(self, aggregator):
        """Equal strength + disagreement → NEUTRAL."""
        tda = pd.DataFrame([
            {"ticker": "X", "direction": "SHORT", "strength": 0.5, "timestamp": "2024-01-02"},
        ])
        nn = pd.DataFrame([
            {"ticker": "X", "direction": "LONG", "strength": 0.5, "timestamp": "2024-01-02"},
        ])
        result = aggregator.aggregate(tda, nn, 0.5, 0.5)
        x_row = result[result["ticker"] == "X"].iloc[0]
        assert x_row["direction"] == "NEUTRAL"
        assert x_row["final_strength"] == 0.0

    def test_only_in_one_strategy(self, aggregator, tda_signals, nn_signals):
        """Tickers only in one strategy should get that direction."""
        result = aggregator.aggregate(tda_signals, nn_signals, 0.5, 0.5)
        # TSLA only in NN
        tsla = result[result["ticker"] == "TSLA"].iloc[0]
        assert tsla["direction"] == "SHORT"

        # GOOG only in TDA as NEUTRAL
        goog = result[result["ticker"] == "GOOG"].iloc[0]
        assert goog["direction"] == "NEUTRAL"

    def test_filter_signals(self, aggregator, tda_signals, nn_signals):
        """filter_signals returns only above threshold, non-neutral."""
        aggregator.aggregate(tda_signals, nn_signals, 0.5, 0.5)
        filtered = aggregator.filter_signals(min_strength=0.4)
        assert (filtered["final_strength"] >= 0.4).all()
        assert (filtered["direction"] != "NEUTRAL").all()

    def test_rank_signals(self, aggregator, tda_signals, nn_signals):
        """rank_signals returns sorted by strength descending."""
        aggregator.aggregate(tda_signals, nn_signals, 0.5, 0.5)
        ranked = aggregator.rank_signals()
        if len(ranked) > 1:
            strengths = ranked["final_strength"].tolist()
            assert strengths == sorted(strengths, reverse=True)

    def test_output_columns(self, aggregator, tda_signals, nn_signals):
        """Output should have required columns."""
        result = aggregator.aggregate(tda_signals, nn_signals, 0.5, 0.5)
        required = {
            "ticker", "direction", "final_strength",
            "tda_component", "nn_component", "agreement", "timestamp",
        }
        assert required.issubset(set(result.columns))

    def test_empty_inputs(self, aggregator):
        """Empty inputs should return empty DataFrame."""
        empty = pd.DataFrame(columns=["ticker", "direction", "strength", "timestamp"])
        result = aggregator.aggregate(empty, empty, 0.5, 0.5)
        assert len(result) == 0

    def test_strength_capped_at_one(self, aggregator):
        """Final strength should never exceed 1.0."""
        tda = pd.DataFrame([
            {"ticker": "X", "direction": "LONG", "strength": 1.0, "timestamp": "2024-01-02"},
        ])
        nn = pd.DataFrame([
            {"ticker": "X", "direction": "LONG", "strength": 1.0, "timestamp": "2024-01-02"},
        ])
        result = aggregator.aggregate(tda, nn, 0.5, 0.5)
        assert result.iloc[0]["final_strength"] <= 1.0
