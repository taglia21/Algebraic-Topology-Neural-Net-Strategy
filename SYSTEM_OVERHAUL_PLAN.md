EXECUTIVE SUMMARY
Current State:

Portfolio: $74,625 (down from $100K starting capital)

Total Drawdown: -25.4%

Peak Portfolio: ~$107K (late Jan/early Feb 2026)

Current Positions: 9 equities, highly correlated (all crypto/growth/tech)

Cash: $199 (0.27% - dangerously overextended)

Active Bot: profit_trader.py (1,001 lines, standalone)

Root Causes Identified:

yfinance data fragility: Using unofficial Yahoo Finance scraper that gets rate-limited/IP-banned

Market order slippage: 100% market orders causing 0.3-3.5% slippage per trade

Fixed-percentage stops: 1.5% stops on volatile names (MARA, MSTR) causing false stop-outs

Position concentration: 9 positions in same sector (crypto/growth), max_correlation_positions=3 config never enforced

Forced EOD close: Closes ALL positions at 4 PM, destroying multi-day winners

Zero cash buffer: 100% capital deployed, no reserves for opportunities or drawdowns

Disconnected infrastructure: Team of Rivals, TDA, ML models all unused

Expected Impact After Fixes:

Transaction costs: -10-15% improvement via limit orders

False stop-outs: -30-50% reduction via ATR stops

Drawdown control: -20-30% via regime awareness and sector caps

Alpha generation: +15-25% via reconnecting TDA/ML/pairs strategies

Overall P&L improvement target: +40-60% over next 3 months

PHASE 1: P0 CRITICAL FIXES (profit_trader.py)
Fix 1: Replace yfinance with Alpaca Data API
Problem:

Line 307-320 in profit_trader.py: MarketData.get_stock_data() uses yf.download()

yfinance is an unofficial scraper with aggressive rate limiting

Users report IP bans after fetching 20-30 tickers at 5min intervals

Bot scans 29 tickers every 120 seconds = guaranteed rate limit failure

Files Affected:

profit_trader.py lines 307-320 (MarketData class)

profit_trader.py lines 140-170 (AlpacaClient already has get_bars() method)

Current Code:

python
class MarketData:
    def __init__(self):
        pass
    
    def get_stock_data(self, ticker: str, period: str = '5d', interval: str = '5m'):
        if not HAS_YF:
            return None
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
            # ...
Solution:

Add client: AlpacaClient parameter to MarketData.__init__()

Refactor get_stock_data() to call self.client.get_bars() first

Convert Alpaca bars response to pandas DataFrame matching current format

Keep yfinance as fallback only if Alpaca fails

Update ProfitTrader.__init__() line 945 to pass self.client to MarketData

Implementation Steps:

python
class MarketData:
    def __init__(self, client: AlpacaClient):
        self.client = client
    
    def get_stock_data(self, ticker: str, period: str = '5d', interval: str = '5m'):
        # Try Alpaca first
        try:
            bars = self.client.get_bars(ticker, timeframe='5Min', limit=390)  # ~5 days of 5min bars
            if bars:
                df = pd.DataFrame(bars)
                df = df.set_index('t')  # timestamp column
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'vw', 'n']
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.debug(f"Alpaca data failed for {ticker}: {e}")
        
        # Fallback to yfinance
        if not HAS_YF:
            return None
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
            # ... rest of existing code
Test Criteria:

Run: python3 profit_trader.py --mode scan and verify no yfinance rate limit errors in logs

Check log shows "Using Alpaca data for [ticker]" for all 29 universe tickers

Verify DataFrame returned has correct shape (>100 rows, 5 columns: OHLCV)

Expected Impact: Eliminates single biggest fragility; prevents trading system failures from data source issues

Fix 2: Switch to Limit Orders
Problem:

Line 228 in profit_trader.py: AlpacaClient.place_order() hardcoded to order_type='market'

Line 881 in ProfitTrader.scan_and_trade(): always calls place_order() with order_type='market'

Market orders cause 0.3-3.5% slippage especially on low-liquidity names (MARA, SMCI, RIOT)

Files Affected:

profit_trader.py lines 195-260 (AlpacaClient.place_order)

profit_trader.py lines 850-900 (ProfitTrader.scan_and_trade execution logic)

Solution:

Modify place_order() to accept order_type parameter (default='limit')

When placing buy orders, fetch current quote via get_snapshot()

Set limit price = ask + $0.02 (or +0.05% for stocks >$100)

Add 30-second fill timeout - if not filled, cancel and replace with market order

Log slippage saved vs market order

Implementation Steps:

python
def place_order(self, symbol: str, qty: int, side: str, order_type: str = 'limit', ...):
    # Fetch current quote for limit price
    if order_type == 'limit' and side == 'buy':
        snapshot = self.get_snapshot(symbol)
        if snapshot and 'latestQuote' in snapshot:
            ask = snapshot['latestQuote']['ap']
            limit_price = ask + (0.02 if ask < 100 else ask * 0.0005)
        else:
            order_type = 'market'  # Fallback if no quote available
    
    # Place limit order with 30sec timeout...
Test Criteria:

Monitor next 10 trades: all should show "ORDER PLACED: BUY X SYMBOL @ limit" in logs

Check fill rate: >95% of limit orders should fill within 30 seconds

Compare limit fill price vs snapshot ask: slippage should be <0.1%

Expected Impact: Save 10-15% on transaction costs; on $67K deployed = ~$6,700-$10,000 annual savings

Fix 3: ATR-Based Dynamic Stop Losses
Problem:

Line 60 in profit_trader.py: stop_loss_pct: float = 0.015 (fixed 1.5%)

Line 618 in TrackedPosition: stores fixed-percentage stop_loss

MARA moves 5-10%/day; 1.5% stop = stopped out on noise

AAPL moves 1-2%/day; 1.5% stop may be too wide

Files Affected:

profit_trader.py lines 558-580 (TrackedPosition dataclass)

profit_trader.py lines 475-525 (SignalGenerator methods computing stops)

profit_trader.py lines 674-750 (PositionManager.check_exits stop logic)

Current Code:

python
def _check_momentum(self, a: Dict) -> Optional[Signal]:
    # ...
    stop = price * (1 - self.config.stop_loss_pct)  # Fixed 1.5%
    target = price * (1 + self.config.take_profit_pct)
Solution:

ATR is ALREADY calculated in analyze() line 405 - use it!

Modify each signal generator method to compute: stop = price - (atr_multiplier * atr)

Momentum/trend: 2.5x ATR

Mean reversion: 1.5x ATR

VWAP: 2.0x ATR

Pass ATR value through Signal dataclass

Update TrackedPosition to store atr value

In check_exits(), use ATR-based stop instead of fixed percentage

Implementation Steps:

python
@dataclass
class Signal:
    ticker: str
    strategy: str
    side: str
    confidence: float
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    atr: float = 0.0  # Add ATR field

def _check_momentum(self, a: Dict) -> Optional[Signal]:
    price = a['price']
    atr = a['atr']
    
    # ATR-based stop
    stop = price - (2.5 * atr)  # 2.5x ATR for momentum
    target = price * (1 + self.config.take_profit_pct)
    
    return Signal(
        ticker=a['ticker'],
        strategy='momentum',
        side='buy',
        confidence=min(conf, 0.95),
        price=price,
        stop_loss=stop,
        take_profit=target,
        reason=f"...",
        atr=atr
    )
Test Criteria:

Log should show "Stop loss: $XX.XX (2.5x ATR=$X.XX)" for each trade entry

Compare stop distance on MARA vs AAPL: MARA stop should be 3-5x wider

Track stop-out rate over 50 trades: should decrease by 30-50% vs fixed stops

Expected Impact: Reduce false stop-outs by 30-50%; let winners run longer on volatile names

Fix 4: Enforce Sector/Correlation Caps
Problem:

Line 77 in profit_trader.py: max_correlation_positions: int = 3 exists in config

NEVER ENFORCED in code - no sector grouping or correlation check

Current live portfolio: 9 positions, 7 are crypto/growth (MARA, COIN, MSTR, HOOD, SMCI, SOFI, ROKU)

All move together = single correlated bet, not diversification

Files Affected:

profit_trader.py lines 30-115 (TraderConfig dataclass - add SECTOR_MAP)

profit_trader.py lines 855-895 (ProfitTrader.scan_and_trade - add sector check before execution)

Solution:

Add SECTOR_MAP constant after TraderConfig definition:

python
SECTOR_MAP = {
    'crypto_adjacent': ['MARA', 'RIOT', 'COIN', 'MSTR', 'HOOD'],
    'mega_tech': ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
    'high_beta_growth': ['PLTR', 'ARM', 'SMCI', 'SOFI', 'SNOW', 'CRM', 'PANW', 'SHOP', 'ROKU', 'UBER'],
    'semis': ['AMD', 'AVGO'],
    'streaming': ['NFLX'],
    'etfs': ['SPY', 'QQQ', 'TQQQ', 'SOXL']
}
Add helper method to ProfitTrader:

python
def get_sector(self, symbol: str) -> str:
    for sector, tickers in SECTOR_MAP.items():
        if symbol in tickers:
            return sector
    return 'other'

def count_positions_in_sector(self, sector: str) -> int:
    count = 0
    for sym in self.positions.tracked.keys():
        if self.get_sector(sym) == sector:
            count += 1
    return count
In scan_and_trade(), before executing signal:

python
for signal in to_execute:
    # Check sector cap
    signal_sector = self.get_sector(signal.ticker)
    if self.count_positions_in_sector(signal_sector) >= self.config.max_correlation_positions:
        logger.info(f"SKIPPED {signal.ticker}: already have {self.config.max_correlation_positions} positions in sector {signal_sector}")
        continue
    
    # Execute trade...
Test Criteria:

Run bot with 9 positions across 3 sectors

Try to enter 4th position in saturated sector

Log should show "SKIPPED TICKER: already have 3 positions in sector CRYPTO_ADJACENT"

Verify portfolio diversifies across 4+ sectors

Expected Impact: Reduce correlated drawdowns by 30-40%; portfolio drawdown should be 30-40% lower than individual position volatility

Fix 5: Maintain 15% Cash Buffer
Problem:

Line 67 in profit_trader.py: position_pct: float = 0.10 (10% per position × 10 positions = 100% deployed)

Current portfolio: $199 cash on $74,625 equity = 0.27% cash

Zero cushion for opportunities, margin calls, or drawdowns

Files Affected:

profit_trader.py lines 30-115 (TraderConfig - add min_cash_pct)

profit_trader.py lines 826-843 (ProfitTrader.calculate_position_size)

Solution:

Add to TraderConfig line 114:

python
min_cash_pct: float = 0.15  # Keep minimum 15% cash buffer
Modify calculate_position_size():

python
def calculate_position_size(self, price: float) -> int:
    acct = self.client.get_account()
    equity = float(acct['equity'])
    cash = float(acct['cash'])
    
    # Use config percentage of equity
    target_value = equity * self.config.position_pct
    
    # Enforce cash buffer
    min_cash = equity * self.config.min_cash_pct
    if cash - target_value < min_cash:
        logger.warning(f"Cash buffer constraint: need ${min_cash:,.0f} cash, have ${cash:,.0f}")
        target_value = max(0, cash - min_cash)  # Reduce position size to preserve buffer
        if target_value < equity * 0.03:  # Less than 3% position not worth it
            logger.warning(f"Skipping trade: insufficient capital for meaningful position")
            return 0
    
    shares = int(target_value / price)
    return max(1, shares) if shares > 0 else 0
Test Criteria:

Portfolio at $75K