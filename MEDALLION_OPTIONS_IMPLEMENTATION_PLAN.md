# 🏗️ MEDALLION-GRADE OPTIONS ENGINE - IMPLEMENTATION PLAN

## Phase 2-6 Complete Build Roadmap

**Start Date**: February 4, 2026  
**Target Completion**: 6-8 weeks  
**Status**: Phase 1 Audit ✅ Complete | Phase 2 Design 🏗️ In Progress

---

## ARCHITECTURE OVERVIEW

```
src/options/
├── __init__.py
├── theta_decay_engine.py       # Core theta optimization
├── iv_analyzer.py              # IV rank/percentile with history
├── greeks_manager.py           # Portfolio Greeks tracking & limits
├── delay_adapter.py            # 15-minute delay compensation
├── position_manager.py         # Position sizing & risk management
├── strategy_engine.py          # Premium selling strategies
├── tradier_executor.py         # Tradier API integration
├── backtest_engine.py          # Historical validation
└── utils/
    ├── black_scholes.py        # Pricing models (from V50)
    ├── risk_metrics.py         # Kelly, Sharpe, etc.
    └── constants.py            # Configuration constants

tests/options/
├── test_theta_decay.py
├── test_iv_analyzer.py
├── test_greeks_manager.py
├── test_delay_adapter.py
├── test_position_manager.py
├── test_strategy_engine.py
├── test_tradier_executor.py
└── test_integration.py
```

---

## PHASE 2: MATHEMATICAL FOUNDATION

### 2.1 Theta Decay Optimization Engine

**File**: `src/options/theta_decay_engine.py`

**Purpose**: Optimize option entry/exit timing based on theta decay acceleration.

**Key Functions**:
```python
class ThetaDecayEngine:
    def calculate_optimal_dte(
        iv_rank: float, 
        underlying_trend: str, 
        volatility_regime: str
    ) -> Tuple[int, int]:
        """
        Returns (entry_dte, exit_dte) based on conditions.
        
        Logic:
        - High IV (>70): Shorter DTE (30-35 days) for faster decay
        - Medium IV (30-70): Standard DTE (35-45 days)
        - Low IV (<30): Longer DTE (45-60 days) or skip
        
        Exit:
        - Take profit at 50% max gain OR
        - Exit at 14-21 DTE (theta acceleration zone)
        """
        
    def calculate_theta_per_day(
        option_price: float,
        strike: float,
        dte: int,
        iv: float
    ) -> float:
        """Calculate expected theta decay per day."""
        
    def project_decay_curve(
        current_dte: int,
        current_price: float,
        target_dte: int
    ) -> pd.DataFrame:
        """Project theta decay over time."""
```

**Theta Decay Science**:
- Theta accelerates exponentially in final 21 days
- Optimal entry: 30-45 DTE (linear decay phase)
- Optimal exit: 14-21 DTE (before acceleration kills you)
- Maximum theta: ATM options (balance risk/reward)

**Performance Target**: Capture 60-80% of theoretical theta

---

### 2.2 IV Analyzer

**File**: `src/options/iv_analyzer.py`

**Purpose**: Calculate IV rank/percentile to identify premium selling opportunities.

**Key Functions**:
```python
class IVAnalyzer:
    def __init__(self, lookback_days: int = 252):
        self.history: Dict[str, deque] = {}  # Symbol -> IV history
        
    def update(self, symbol: str, current_iv: float) -> None:
        """Add new IV observation to rolling window."""
        
    def get_iv_rank(self, symbol: str, current_iv: float) -> float:
        """
        IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
        
        Returns 0-100 score.
        >70 = High IV (sell premium)
        <30 = Low IV (buy premium or skip)
        """
        
    def get_iv_percentile(self, symbol: str, current_iv: float) -> float:
        """
        IV Percentile = % of days where IV was below current level
        
        More stable than IV Rank (not affected by outliers).
        """
        
    def calculate_hv_iv_ratio(
        symbol: str,
        historical_vol: float,
        implied_vol: float
    ) -> float:
        """
        HV/IV Ratio:
        >1.0 = IV underpriced (buy options)
        <1.0 = IV overpriced (sell premium)
        """
        
    def detect_iv_regime(self, symbol: str) -> str:
        """Classify as: 'low', 'normal', 'elevated', 'extreme'."""
```

**Data Storage**:
- Use `collections.deque` with maxlen for memory efficiency
- Persist to JSON/SQLite for historical analysis
- Load on startup from saved state

**Edge Detection**:
- IV Rank >50 AND HV/IV <0.8 = Strong sell signal
- IV expanding rapidly = Earnings/news (avoid)
- IV crushing post-event = Opportunity (if not caught in it)

---

### 2.3 Greeks Manager

**File**: `src/options/greeks_manager.py`

**Purpose**: Track and enforce portfolio-level Greek limits.

**Key Functions**:
```python
class GreeksManager:
    def __init__(self, config: GreeksConfig):
        self.max_portfolio_delta = config.max_delta  # ±0.20 per $10K
        self.max_portfolio_gamma = config.max_gamma
        self.max_portfolio_vega = config.max_vega
        self.target_theta = config.target_theta  # Positive (selling premium)
        
    def calculate_position_greeks(
        position: OptionsPosition
    ) -> Dict[str, float]:
        """Calculate all Greeks for a single position."""
        
    def calculate_portfolio_greeks(
        positions: List[OptionsPosition]
    ) -> PortfolioGreeks:
        """Aggregate Greeks across all positions."""
        
    def check_limits(
        proposed_position: OptionsPosition,
        current_positions: List[OptionsPosition]
    ) -> Tuple[bool, List[str]]:
        """
        Check if adding position would violate limits.
        
        Returns:
            (allowed: bool, violations: List[str])
        """
        
    def suggest_hedge(
        current_greeks: PortfolioGreeks
    ) -> Optional[HedgeSuggestion]:
        """
        If limits breached, suggest hedge:
        - High delta -> opposite position
        - High gamma -> gamma scalp
        - High vega -> volatility hedge
        """
        
    def update_greeks_realtime(
        positions: List[OptionsPosition],
        market_data: MarketData
    ) -> None:
        """Recalculate Greeks as underlying moves."""
```

**Portfolio Limits** (per $100K capital):
```python
MAX_PORTFOLIO_DELTA = 20.0    # ±$2,000 per $1 underlying move
MAX_PORTFOLIO_GAMMA = 5.0     # Control acceleration risk
MAX_PORTFOLIO_VEGA = -50.0    # Negative (short vol), max $50 per 1% IV
TARGET_THETA = 50.0           # Earn $50/day from decay (goal)
MAX_POSITIONS = 6             # Concentration limit
```

**Monitoring**:
- Recalculate on every price update
- Alert if >80% of limit
- Auto-hedge if >100% of limit (emergency)

---

## PHASE 3: 15-MINUTE DELAY ADAPTATION

### 3.1 Delay Adapter

**File**: `src/options/delay_adapter.py`

**Purpose**: Compensate for stale pricing in Tradier sandbox.

**Key Functions**:
```python
class DelayAdapter:
    def __init__(self, delay_minutes: int = 15):
        self.delay = delay_minutes
        self.volatility_cache = {}
        
    def calculate_price_uncertainty(
        symbol: str,
        current_price: float,
        atr_14: float,
        vix_level: float
    ) -> PriceUncertainty:
        """
        Estimate price movement over delay period.
        
        Returns:
            - expected_range: (low, high)
            - std_dev: standard deviation
            - confidence: 95% confidence interval
        """
        # Use ATR to estimate 15-min movement
        # minute_vol = daily_vol / sqrt(390 trading minutes)
        # movement_15min = minute_vol * sqrt(15) * current_price
        
    def adjust_entry_price(
        displayed_price: float,
        spread: float,
        volatility: float,
        order_side: str  # 'buy' or 'sell'
    ) -> float:
        """
        Add safety buffer to entry price.
        
        For selling premium (credit):
            - Use bid - 1 std_dev (worse price)
        For buying premium (debit):
            - Use ask + 1 std_dev (worse fill)
        """
        
    def adjust_greeks(
        greeks: Greeks,
        time_delay: int,
        price_uncertainty: float
    ) -> Greeks:
        """
        Adjust Greeks for staleness.
        
        Delta/Gamma may have changed significantly.
        Use conservative estimates.
        """
        
    def is_safe_entry_window(
        current_time: datetime,
        vix_level: float
    ) -> Tuple[bool, str]:
        """
        Determine if it's safe to enter with delayed data.
        
        Avoid:
        - First 30 min (9:30-10:00 AM ET)
        - Last 30 min (3:30-4:00 PM ET)
        - VIX >30 (high volatility)
        - FOMC days
        - Major earnings announcements
        
        Returns:
            (is_safe: bool, reason: str)
        """
```

**Safety Buffers**:
```python
# For credit spreads (selling premium)
CREDIT_BUFFER = 1.5  # Use 1.5 std dev buffer (worse fill)

# For debit spreads (buying)
DEBIT_BUFFER = 2.0  # Use 2.0 std dev buffer (extra conservative)

# Time windows (Eastern Time)
SAFE_WINDOWS = [
    (time(10, 0), time(11, 30)),  # Morning
    (time(13, 0), time(15, 0)),   # Afternoon
]

# Maximum VIX for entry
MAX_VIX_FOR_ENTRY = 35.0
```

**Example**:
```python
# Displayed: SPY $450, bid $2.00, ask $2.10
# Real-time: SPY might be $450.50, bid $2.15, ask $2.25
# 
# If selling put credit spread:
# - Don't use displayed $2.00 credit
# - Expect only $1.80 credit (20¢ buffer)
# - Still profitable? Proceed. Otherwise skip.
```

---

## PHASE 4: PREMIUM SELLING STRATEGIES

### 4.1 The Wheel Strategy

**File**: `src/options/strategy_engine.py`

```python
class WheelStrategy:
    """
    Step 1: Sell cash-secured put (CSP)
    Step 2: If assigned, own stock, sell covered call
    Step 3: If called away, return to Step 1
    
    Repeat indefinitely (hence "Wheel")
    """
    
    def select_put_strike(
        underlying_price: float,
        iv_rank: float,
        dte_range: Tuple[int, int] = (30, 45)
    ) -> OptionContract:
        """
        Select optimal put to sell.
        
        Criteria:
        - Delta: 0.20-0.30 (70-80% probability OTM)
        - Strike: ~3-7% below current price
        - DTE: 30-45 days
        - Premium: >1% of strike per month
        - IV Rank: >50 (only in high IV)
        """
        
    def select_call_strike(
        cost_basis: float,
        underlying_price: float,
        dte_range: Tuple[int, int] = (30, 45)
    ) -> OptionContract:
        """
        Select covered call to sell.
        
        Criteria:
        - Delta: 0.25-0.35
        - Strike: Above cost basis (don't sell at loss)
        - Premium: >1% of stock price per month
        """
        
    def manage_position(
        position: WheelPosition,
        current_price: float,
        current_dte: int
    ) -> Optional[str]:
        """
        Decide if action needed.
        
        Actions:
        - 'close_profit': Hit 50% profit target
        - 'close_time': At 21 DTE, close to avoid gamma
        - 'roll': Roll to next expiration
        - 'hold': Continue holding
        """
```

**Expected Performance**:
- Win rate: 70-80% (most expire worthless)
- Annual return: 12-20% on capital
- Max drawdown: 15-25% (during assignments)
- Sharpe: 0.8-1.2

---

### 4.2 Credit Spread Strategy

```python
class CreditSpreadStrategy:
    """
    Defined-risk alternative to naked options.
    
    Bull Put Spread: Sell put, buy lower put (bullish)
    Bear Call Spread: Sell call, buy higher call (bearish)
    """
    
    def construct_bull_put_spread(
        underlying_price: float,
        iv_rank: float,
        max_risk: float
    ) -> Spread:
        """
        Build bull put spread.
        
        Structure:
        - Sell put at ~0.30 delta (70% POP)
        - Buy put 5-10 points lower (protection)
        - Collect 1/3 to 1/2 of spread width
        
        Example:
        SPY at $450
        - Sell $440 put for $2.50
        - Buy $435 put for $0.80
        - Net credit: $1.70
        - Max profit: $170 (keep credit)
        - Max loss: $330 ($5 width - $1.70 credit)
        - Risk/Reward: 1.94 (good)
        """
        
    def calculate_spread_metrics(
        spread: Spread
    ) -> SpreadMetrics:
        """
        Calculate:
        - Max profit / Max loss
        - Probability of profit
        - Expected value
        - Breakeven price
        - Return on risk
        """
        
    def manage_spread(
        position: SpreadPosition,
        current_price: float,
        current_pnl: float
    ) -> str:
        """
        Management rules:
        - Take profit: 50% of max gain
        - Stop loss: 2x credit received (200% loss)
        - Time exit: 21 DTE regardless of P&L
        - Adjustment: If breached, consider roll
        """
```

**Expected Performance**:
- Win rate: 65-75%
- Annual return: 15-25%
- Max drawdown: 10-20%
- Sharpe: 1.0-1.5

---

### 4.3 Iron Condor Strategy

```python
class IronCondorStrategy:
    """
    Delta-neutral income strategy.
    
    Structure:
    - Sell OTM call spread (above market)
    - Sell OTM put spread (below market)
    - Collect premium from both sides
    - Profit if price stays in range
    """
    
    def construct_iron_condor(
        underlying_price: float,
        iv_rank: float,
        expected_move: float,
        capital: float
    ) -> IronCondor:
        """
        Build iron condor.
        
        Strikes (for SPY at $450):
        - Sell $460 call / Buy $465 call (call spread)
        - Sell $440 put / Buy $435 put (put spread)
        
        Credit collected: $2.00 total
        Max profit: $200
        Max loss: $300 (on either side)
        
        Probability of profit: ~60-70%
        """
        
    def calculate_expected_move(
        underlying_price: float,
        iv: float,
        dte: int
    ) -> float:
        """
        Calculate 1 std dev move.
        
        Formula: price * IV * sqrt(DTE/365)
        
        Example: $450 * 0.20 * sqrt(45/365) = $31.5
        
        Place short strikes outside expected move.
        """
        
    def manage_iron_condor(
        position: IronCondorPosition,
        current_price: float,
        current_delta: float
    ) -> str:
        """
        Management:
        - Profit target: 50% of max
        - Adjustment trigger: Delta >|0.15| on one side
        - Time exit: 21 DTE
        - Stop loss: If one side 2x credit
        """
```

**Expected Performance**:
- Win rate: 60-70%
- Annual return: 20-35%
- Max drawdown: 15-25%
- Sharpe: 1.2-2.0

---

## PHASE 5: RISK MANAGEMENT

### 5.1 Position Sizing

**File**: `src/options/position_manager.py`

```python
class PositionManager:
    def calculate_position_size(
        strategy: Strategy,
        max_loss_per_contract: float,
        account_size: float,
        iv_environment: float,
        confidence: float
    ) -> int:
        """
        Calculate contracts to trade.
        
        Base formula:
        risk_per_trade = account_size * 0.02  # 2% max
        
        Adjustments:
        - IV environment: Reduce size in high IV (more risk)
        - Confidence: Scale with signal strength
        - Delay buffer: 20% reduction for stale data
        - Portfolio correlation: Reduce if similar positions
        
        Example:
        - Account: $100,000
        - Base risk: $2,000 (2%)
        - Max loss per contract: $300
        - IV adjustment: 0.8 (high IV)
        - Delay adjustment: 0.8
        - Contracts: $2000 * 0.8 * 0.8 / $300 = 4 contracts
        """
        
    def check_buying_power(
        proposed_position: OptionsPosition,
        current_positions: List[OptionsPosition],
        account_balance: float
    ) -> Tuple[bool, float]:
        """
        Verify sufficient capital.
        
        Returns:
            (has_capital: bool, available: float)
        """
```

---

### 5.2 Exit Management

```python
class ExitManager:
    def generate_exit_signal(
        position: OptionsPosition,
        current_price: float,
        current_dte: int,
        current_pnl: float
    ) -> Optional[ExitSignal]:
        """
        Check exit conditions.
        
        Exits:
        1. Profit target (50% of max)
        2. Stop loss (2x credit for premium selling)
        3. Time stop (21 DTE)
        4. Greeks breach (delta too high)
        5. IV crush/spike (exit opportunity)
        """
        
    def calculate_adjustment(
        position: OptionsPosition,
        breach_side: str
    ) -> Optional[Adjustment]:
        """
        If position breached, suggest:
        - Roll out (same strike, later date)
        - Roll out and down/up (reduce risk)
        - Close and re-enter
        - Take loss and move on
        """
```

---

## PHASE 6: IMPLEMENTATION & TESTING

### 6.1 File Structure Creation

Will create all files in proper structure with:
- Type hints throughout
- Comprehensive docstrings
- Logging at all levels
- Error handling
- Input validation

### 6.2 Testing Strategy

**Unit Tests** (90%+ coverage target):
- Each function tested independently
- Edge cases covered
- Mock external dependencies (API calls)

**Integration Tests**:
- Full workflow end-to-end
- Tradier API integration (sandbox)
- Position lifecycle

**Backtesting**:
- Historical options data (will need to acquire)
- Strategy performance validation
- Risk metrics calculation

### 6.3 Documentation

- README for options module
- Strategy guides for each approach
- API reference
- Configuration guide
- Deployment checklist

---

## SUCCESS METRICS (Validation Criteria)

Before declaring "production ready":

✅ **Code Quality**:
- [ ] 90%+ test coverage
- [ ] No critical linting errors
- [ ] Type hints throughout
- [ ] Comprehensive logging

✅ **Functionality**:
- [ ] Can fetch options chain
- [ ] Can calculate Greeks accurately
- [ ] Can size positions correctly
- [ ] Can place orders (dry run)
- [ ] Can track positions
- [ ] Can generate exits

✅ **Risk Controls**:
- [ ] Position size limits enforced
- [ ] Portfolio Greeks limits enforced
- [ ] Stop losses implemented
- [ ] Profit targets implemented
- [ ] Time exits implemented

✅ **Performance** (Backtested):
- [ ] Win rate >65% (premium selling)
- [ ] Sharpe ratio >1.0
- [ ] Max drawdown <15%
- [ ] Profit factor >1.5

✅ **Production Readiness**:
- [ ] Error handling comprehensive
- [ ] Logging production-grade
- [ ] Configuration externalized
- [ ] Documentation complete
- [ ] Deployment guide written

---

## TIMELINE

**Week 1-2**: Core engines (theta, IV, Greeks)
**Week 3-4**: Delay adapter, position manager
**Week 5-6**: Strategy implementations
**Week 7-8**: Testing, backtesting, documentation

**Total**: 6-8 weeks to production-ready system

---

## NEXT ACTIONS

1. ✅ Review and approve this plan
2. 🏗️ Begin implementation (starting now)
3. 📊 Acquire historical options data
4. 🧪 Set up testing infrastructure
5. 🚀 Deploy to paper trading
6. 📈 Monitor and iterate

---

**Status**: Ready to begin implementation  
**Confidence**: High (solid foundation from V50)  
**Risk**: Medium (need historical data for validation)

Let's build this! 🚀
