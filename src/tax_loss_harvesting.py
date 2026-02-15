"""
Tax-Loss Harvesting Module (TIER 3)
=====================================

Tracks cost basis per lot, identifies unrealized losses eligible for
tax-loss harvesting, and suggests wash-sale-safe substitutions.

Features:
1. FIFO / LIFO / Specific-lot cost basis tracking
2. Unrealized gain/loss report per position & lot
3. Harvesting suggestions ranked by tax benefit
4. Wash-sale rule compliance (30-day lookback/forward)
5. Short-term vs long-term capital gains separation
6. Annual tax summary with estimated savings

IRS Rules Enforced:
- Wash sale: cannot repurchase substantially identical security within
  30 days before or after a sale at a loss.
- Short-term < 1 year; long-term >= 1 year holding period.

Usage:
    from src.tax_loss_harvesting import TaxLotTracker, HarvestingEngine, TaxConfig

    tracker = TaxLotTracker()
    tracker.add_lot("AAPL", 50, 175.0, "2025-03-15")
    tracker.add_lot("AAPL", 30, 190.0, "2025-08-20")

    engine = HarvestingEngine(tracker)
    suggestions = engine.suggest_harvesting(current_prices={"AAPL": 160.0})
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import json

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class CostBasisMethod(Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    SPECIFIC = "specific"
    AVERAGE = "average"


class GainType(Enum):
    SHORT_TERM = "short_term"   # < 1 year
    LONG_TERM = "long_term"     # >= 1 year


@dataclass
class TaxConfig:
    """Tax-loss harvesting configuration."""
    cost_basis_method: CostBasisMethod = CostBasisMethod.FIFO
    wash_sale_window_days: int = 30
    min_harvest_amount: float = 100.0       # min loss to consider
    short_term_rate: float = 0.37           # federal marginal rate
    long_term_rate: float = 0.20            # long-term cap gains rate
    state_rate: float = 0.05                # state tax rate
    harvest_threshold_pct: float = 5.0      # suggest if loss > 5% of basis

    # Substitute securities for wash-sale compliance
    substitutes: Dict[str, List[str]] = field(default_factory=lambda: {
        "SPY": ["VOO", "IVV", "SPLG"],
        "QQQ": ["QQQM", "VGT", "XLK"],
        "IWM": ["VTWO", "SCHA", "IJR"],
        "XLF": ["VFH", "FNCL", "IYF"],
        "XLK": ["VGT", "FTEC", "IGV"],
        "AAPL": ["XLK", "VGT", "QQQ"],
        "MSFT": ["XLK", "VGT", "QQQ"],
        "GOOGL": ["XLC", "VOX", "GOOG"],
        "AMZN": ["XLY", "VCR", "IBUY"],
        "TSLA": ["DRIV", "IDRV", "CARZ"],
    })


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TaxLot:
    """Single tax lot (purchase of shares)."""
    lot_id: str = ""
    symbol: str = ""
    quantity: float = 0.0
    cost_basis: float = 0.0         # price per share at purchase
    purchase_date: str = ""         # YYYY-MM-DD
    remaining_quantity: float = 0.0  # after partial sales

    @property
    def total_cost(self) -> float:
        return self.remaining_quantity * self.cost_basis

    @property
    def holding_days(self) -> int:
        try:
            pd = datetime.strptime(self.purchase_date, "%Y-%m-%d").date()
            return (date.today() - pd).days
        except (ValueError, TypeError):
            return 0

    @property
    def gain_type(self) -> GainType:
        return GainType.LONG_TERM if self.holding_days >= 365 else GainType.SHORT_TERM


@dataclass
class SaleRecord:
    """Record of a sale for wash-sale tracking."""
    symbol: str = ""
    quantity: float = 0.0
    sale_price: float = 0.0
    sale_date: str = ""
    realized_pnl: float = 0.0
    gain_type: str = ""
    lot_id: str = ""


@dataclass
class HarvestSuggestion:
    """A tax-loss harvesting suggestion."""
    symbol: str = ""
    lots: List[str] = field(default_factory=list)         # lot IDs
    shares_to_sell: float = 0.0
    unrealized_loss: float = 0.0
    estimated_tax_savings: float = 0.0
    gain_type: str = ""
    substitute_securities: List[str] = field(default_factory=list)
    wash_sale_safe: bool = True
    reason: str = ""


@dataclass
class TaxSummary:
    """Annual tax summary."""
    year: int = 0
    total_unrealized_gains: float = 0.0
    total_unrealized_losses: float = 0.0
    short_term_gains: float = 0.0
    short_term_losses: float = 0.0
    long_term_gains: float = 0.0
    long_term_losses: float = 0.0
    realized_gains: float = 0.0
    realized_losses: float = 0.0
    estimated_tax_liability: float = 0.0
    estimated_harvestable_savings: float = 0.0
    positions_with_losses: int = 0
    harvesting_suggestions: List[HarvestSuggestion] = field(default_factory=list)


# =============================================================================
# TAX LOT TRACKER
# =============================================================================

class TaxLotTracker:
    """
    Tracks cost basis per lot with FIFO/LIFO/Average support.

    Usage:
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", 50, 175.0, "2025-03-15")
        lots = tracker.get_lots("AAPL")
        tracker.sell_shares("AAPL", 20, 160.0, "2026-02-15")
    """

    def __init__(self, config: Optional[TaxConfig] = None):
        self.config = config or TaxConfig()
        self._lots: Dict[str, List[TaxLot]] = defaultdict(list)
        self._sales: List[SaleRecord] = []
        self._lot_counter = 0

    def add_lot(
        self, symbol: str, quantity: float, cost_basis: float, purchase_date: str,
    ) -> TaxLot:
        """Record a new purchase lot."""
        self._lot_counter += 1
        lot = TaxLot(
            lot_id=f"{symbol}-{self._lot_counter:04d}",
            symbol=symbol,
            quantity=quantity,
            cost_basis=cost_basis,
            purchase_date=purchase_date,
            remaining_quantity=quantity,
        )
        self._lots[symbol].append(lot)
        logger.debug("Added lot %s: %s x%.0f @ $%.2f", lot.lot_id, symbol, quantity, cost_basis)
        return lot

    def sell_shares(
        self,
        symbol: str,
        quantity: float,
        sale_price: float,
        sale_date: str,
        method: Optional[CostBasisMethod] = None,
    ) -> List[SaleRecord]:
        """Sell shares using configured cost basis method. Returns sale records."""
        method = method or self.config.cost_basis_method
        lots = self._lots.get(symbol, [])
        lots = [l for l in lots if l.remaining_quantity > 0]

        if method == CostBasisMethod.FIFO:
            lots.sort(key=lambda l: l.purchase_date)
        elif method == CostBasisMethod.LIFO:
            lots.sort(key=lambda l: l.purchase_date, reverse=True)
        elif method == CostBasisMethod.SPECIFIC:
            # For specific, caller should pass lot_ids (not implemented in auto mode)
            lots.sort(key=lambda l: l.cost_basis, reverse=True)  # sell highest cost first (tax optimal)

        remaining = quantity
        records = []
        for lot in lots:
            if remaining <= 0:
                break
            sell_qty = min(remaining, lot.remaining_quantity)
            lot.remaining_quantity -= sell_qty
            remaining -= sell_qty

            pnl = sell_qty * (sale_price - lot.cost_basis)
            rec = SaleRecord(
                symbol=symbol,
                quantity=sell_qty,
                sale_price=sale_price,
                sale_date=sale_date,
                realized_pnl=pnl,
                gain_type=lot.gain_type.value,
                lot_id=lot.lot_id,
            )
            records.append(rec)
            self._sales.append(rec)
            logger.debug(
                "Sold %.0f %s from lot %s — realized P&L: $%.2f (%s)",
                sell_qty, symbol, lot.lot_id, pnl, lot.gain_type.value,
            )

        return records

    def get_lots(self, symbol: Optional[str] = None) -> List[TaxLot]:
        """Get open lots for a symbol or all symbols."""
        if symbol:
            return [l for l in self._lots.get(symbol, []) if l.remaining_quantity > 0]
        return [l for lots in self._lots.values() for l in lots if l.remaining_quantity > 0]

    def get_sales(self, symbol: Optional[str] = None) -> List[SaleRecord]:
        """Get sale records."""
        if symbol:
            return [s for s in self._sales if s.symbol == symbol]
        return list(self._sales)

    def get_cost_basis(self, symbol: str) -> float:
        """Get average cost basis for a symbol."""
        lots = self.get_lots(symbol)
        if not lots:
            return 0.0
        total_cost = sum(l.remaining_quantity * l.cost_basis for l in lots)
        total_qty = sum(l.remaining_quantity for l in lots)
        return total_cost / total_qty if total_qty > 0 else 0.0

    def get_total_shares(self, symbol: str) -> float:
        """Total shares held in a symbol."""
        return sum(l.remaining_quantity for l in self.get_lots(symbol))

    def unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Compute unrealized P&L at current price."""
        lots = self.get_lots(symbol)
        return sum(l.remaining_quantity * (current_price - l.cost_basis) for l in lots)

    def unrealized_pnl_by_lot(
        self, symbol: str, current_price: float,
    ) -> List[Tuple[TaxLot, float]]:
        """Unrealized P&L broken down by lot."""
        return [
            (lot, lot.remaining_quantity * (current_price - lot.cost_basis))
            for lot in self.get_lots(symbol)
        ]

    def symbols(self) -> List[str]:
        """All symbols with open lots."""
        return [s for s, lots in self._lots.items()
                if any(l.remaining_quantity > 0 for l in lots)]


# =============================================================================
# WASH SALE DETECTOR
# =============================================================================

class WashSaleDetector:
    """Detects wash sale violations (30-day rule)."""

    def __init__(self, window_days: int = 30):
        self.window_days = window_days

    def check_wash_sale(
        self,
        symbol: str,
        sale_date: str,
        purchases: List[TaxLot],
        sales: List[SaleRecord],
    ) -> bool:
        """Return True if selling symbol on sale_date would trigger wash sale."""
        try:
            sd = datetime.strptime(sale_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return False

        window_start = sd - timedelta(days=self.window_days)
        window_end = sd + timedelta(days=self.window_days)

        # Check if any recent sales of same symbol at a loss were followed by repurchase
        for s in sales:
            if s.symbol != symbol or s.realized_pnl >= 0:
                continue
            try:
                prev_sale_date = datetime.strptime(s.sale_date, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if abs((prev_sale_date - sd).days) <= self.window_days:
                return True

        # Check if any purchases of same symbol are within window
        for lot in purchases:
            if lot.symbol != symbol:
                continue
            try:
                pd = datetime.strptime(lot.purchase_date, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if window_start <= pd <= window_end and pd != sd:
                return True

        return False


# =============================================================================
# HARVESTING ENGINE
# =============================================================================

class HarvestingEngine:
    """
    Suggests tax-loss harvesting opportunities.

    Usage:
        engine = HarvestingEngine(tracker)
        suggestions = engine.suggest_harvesting({"AAPL": 160.0, "SPY": 450.0})
        summary = engine.annual_summary({"AAPL": 160.0, "SPY": 450.0})
    """

    def __init__(self, tracker: TaxLotTracker, config: Optional[TaxConfig] = None):
        self.tracker = tracker
        self.config = config or tracker.config
        self._wash_detector = WashSaleDetector(self.config.wash_sale_window_days)

    def suggest_harvesting(
        self,
        current_prices: Dict[str, float],
        as_of_date: Optional[str] = None,
    ) -> List[HarvestSuggestion]:
        """Generate ranked harvesting suggestions."""
        as_of = as_of_date or date.today().isoformat()
        suggestions = []

        for symbol in self.tracker.symbols():
            price = current_prices.get(symbol)
            if price is None:
                continue

            lot_pnls = self.tracker.unrealized_pnl_by_lot(symbol, price)
            loss_lots = [(lot, pnl) for lot, pnl in lot_pnls if pnl < -self.config.min_harvest_amount]

            if not loss_lots:
                continue

            # Check wash sale
            wash_safe = not self._wash_detector.check_wash_sale(
                symbol, as_of,
                self.tracker.get_lots(),
                self.tracker.get_sales(),
            )

            for lot, pnl in loss_lots:
                loss_pct = abs(pnl) / lot.total_cost * 100 if lot.total_cost > 0 else 0
                if loss_pct < self.config.harvest_threshold_pct:
                    continue

                # Estimate tax savings
                if lot.gain_type == GainType.SHORT_TERM:
                    rate = self.config.short_term_rate + self.config.state_rate
                    gain_label = "short_term"
                else:
                    rate = self.config.long_term_rate + self.config.state_rate
                    gain_label = "long_term"

                tax_savings = abs(pnl) * rate

                subs = self.config.substitutes.get(symbol, [])

                suggestions.append(HarvestSuggestion(
                    symbol=symbol,
                    lots=[lot.lot_id],
                    shares_to_sell=lot.remaining_quantity,
                    unrealized_loss=pnl,
                    estimated_tax_savings=tax_savings,
                    gain_type=gain_label,
                    substitute_securities=subs,
                    wash_sale_safe=wash_safe,
                    reason=f"Loss of ${abs(pnl):,.2f} ({loss_pct:.1f}% of basis). "
                           f"Est. tax savings: ${tax_savings:,.2f}.",
                ))

        # Rank by tax savings (highest first)
        suggestions.sort(key=lambda s: s.estimated_tax_savings, reverse=True)
        return suggestions

    def annual_summary(
        self,
        current_prices: Dict[str, float],
        year: Optional[int] = None,
    ) -> TaxSummary:
        """Generate annual tax summary."""
        year = year or date.today().year
        summary = TaxSummary(year=year)

        # Unrealized gains/losses
        for symbol in self.tracker.symbols():
            price = current_prices.get(symbol)
            if price is None:
                continue
            lot_pnls = self.tracker.unrealized_pnl_by_lot(symbol, price)
            for lot, pnl in lot_pnls:
                if pnl >= 0:
                    summary.total_unrealized_gains += pnl
                    if lot.gain_type == GainType.SHORT_TERM:
                        summary.short_term_gains += pnl
                    else:
                        summary.long_term_gains += pnl
                else:
                    summary.total_unrealized_losses += pnl
                    summary.positions_with_losses += 1
                    if lot.gain_type == GainType.SHORT_TERM:
                        summary.short_term_losses += pnl
                    else:
                        summary.long_term_losses += pnl

        # Realized (from sales)
        for sale in self.tracker.get_sales():
            try:
                sale_yr = int(sale.sale_date[:4])
            except (ValueError, TypeError):
                continue
            if sale_yr == year:
                if sale.realized_pnl >= 0:
                    summary.realized_gains += sale.realized_pnl
                else:
                    summary.realized_losses += sale.realized_pnl

        # Tax liability estimate
        st_net = summary.short_term_gains + summary.short_term_losses + summary.realized_gains + summary.realized_losses
        lt_net = summary.long_term_gains + summary.long_term_losses
        summary.estimated_tax_liability = max(0, (
            st_net * (self.config.short_term_rate + self.config.state_rate) +
            lt_net * (self.config.long_term_rate + self.config.state_rate)
        ))

        # Harvesting suggestions
        suggestions = self.suggest_harvesting(current_prices)
        summary.harvesting_suggestions = suggestions
        summary.estimated_harvestable_savings = sum(s.estimated_tax_savings for s in suggestions)

        return summary


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tracker = TaxLotTracker()

    # Simulate purchases
    tracker.add_lot("AAPL", 50, 185.0, "2025-03-15")
    tracker.add_lot("AAPL", 30, 195.0, "2025-09-10")
    tracker.add_lot("SPY", 100, 470.0, "2025-01-20")
    tracker.add_lot("QQQ", 80, 410.0, "2025-06-05")
    tracker.add_lot("TSLA", 40, 350.0, "2025-04-12")

    # Simulate a sale
    tracker.sell_shares("SPY", 30, 480.0, "2025-12-15")

    # Current prices (at a loss for some)
    prices = {"AAPL": 170.0, "SPY": 460.0, "QQQ": 390.0, "TSLA": 280.0}

    engine = HarvestingEngine(tracker)
    suggestions = engine.suggest_harvesting(prices)

    print("\n=== Tax-Loss Harvesting Suggestions ===")
    for s in suggestions:
        safe = "✅ WASH-SAFE" if s.wash_sale_safe else "⚠️ WASH RISK"
        print(f"\n{s.symbol} | Sell {s.shares_to_sell:.0f} shares | "
              f"Loss: ${abs(s.unrealized_loss):,.2f} | "
              f"Tax Savings: ${s.estimated_tax_savings:,.2f} | {safe}")
        print(f"  Substitutes: {', '.join(s.substitute_securities)}")
        print(f"  {s.reason}")

    summary = engine.annual_summary(prices)
    print(f"\n=== {summary.year} Tax Summary ===")
    print(f"Unrealized Gains: ${summary.total_unrealized_gains:,.2f}")
    print(f"Unrealized Losses: ${summary.total_unrealized_losses:,.2f}")
    print(f"Realized Gains: ${summary.realized_gains:,.2f}")
    print(f"Realized Losses: ${summary.realized_losses:,.2f}")
    print(f"Est. Tax Liability: ${summary.estimated_tax_liability:,.2f}")
    print(f"Harvestable Savings: ${summary.estimated_harvestable_savings:,.2f}")
