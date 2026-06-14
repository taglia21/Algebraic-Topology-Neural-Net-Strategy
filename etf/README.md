# ETF Tactical-Allocation Engine

A self-contained, **ETF-only** quantitative trading engine for Interactive
Brokers. It is independent of the equities (Alpaca) and VRP (options) engines —
trading it on/off changes nothing in those programs.

## What it does

A monthly-rebalanced tactical asset-allocation strategy over a diversified,
highly liquid multi-asset ETF universe (US/intl equity, sectors, Treasuries,
credit, gold, commodities). It combines only **durable, out-of-sample-robust**
published edges, deliberately avoiding fragile over-parameterised signals:

1. **Trend filter** (Faber 2007; Moskowitz-Ooi-Pedersen 2012) — hold an ETF
   only when price > 200-day SMA *and* 12-1 month momentum > 0.
2. **Cross-sectional / dual momentum** (Antonacci) — rank eligible ETFs, hold
   the top-K with positive absolute momentum.
3. **Inverse-volatility weighting** (risk parity lite) — size by risk so no
   single ETF dominates.
4. **Portfolio volatility targeting** (Barroso-Santa-Clara 2015) — scale gross
   exposure to a constant risk budget; long-only, no leverage by default.
5. **Drawdown de-risking overlay** — progressively cut exposure to cash during
   equity drawdowns.

Capital not allocated to risk parks in a T-bill ETF (`BIL`).

## Honest performance (real data, 2007-01-03 → 2026-06-10)

Out-of-sample in the sense that the strategy uses **no fitted parameters** —
every constant is set from published research, not optimised on this sample.

| Metric            | ETF Engine | SPY (buy & hold) |
|-------------------|-----------:|-----------------:|
| CAGR              |      6.83% |           10.80% |
| Annual volatility |     10.49% |               -- |
| Sharpe            |       0.40 |             0.47 |
| Sortino           |       0.49 |               -- |
| **Max drawdown**  | **-17.86%**|        **-55.19%**|
| Calmar            |       0.38 |               -- |
| Alpha (ann.)      |      4.15% |               -- |
| Beta              |       0.25 |               -- |

**Interpretation (no hype):** this is a *defensive, crisis-resilient* allocator,
not an alpha machine. It beat SPY in the 2008 GFC (+2.8% vs -5.6%) and 2022
bear (-4.0% vs -18.8%) with ~3× smaller drawdowns, but lags badly in strong
bull markets (2010-2021). On a full-cycle, risk-adjusted basis it is roughly
SPY-like in Sharpe with dramatically lower tail risk.

It **does not** clear the institutional Research→Paper promotion gate
(Sharpe ≥ 1.10). Parameter sweeps confirm raising the vol target *reduces*
Sharpe — the risky sleeve lacks enough edge to scale up. Treat this as a solid,
production-shaped **foundation** that needs further research (see roadmap) before
risking meaningful capital.

## Usage

```bash
# Historical backtest (downloads + caches data via yfinance)
python -m etf.main --mode backtest --start 2007-01-01 --out results.json

# Today's target weights (no trading)
python -m etf.main --mode signal

# IBKR paper account — plan orders (dry-run, nothing submitted)
python -m etf.main --mode paper

# IBKR paper — actually submit orders
python -m etf.main --mode paper --execute

# IBKR live — guarded behind two explicit flags
python -m etf.main --mode live --execute --i-understand-the-risk
```

### IBKR connection

Set environment variables (defaults target IB Gateway paper on `4002`):

```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002          # 4001 live GW, 7497 TWS paper, 7496 TWS live
export ETF_IBKR_CLIENT_ID=7    # distinct from the VRP engine's client id
export IBKR_ACCOUNT=DUxxxxxxx
```

Requires `ib_async` (`pip install ib_async`). The broker runs **dry-run by
default** and fails safe (aborts the rebalance) on any missing price/account data.

## Key configuration (`etf/config.py`)

| Knob | Env var | Default | Rationale |
|------|---------|--------:|-----------|
| Target volatility | `ETF_TARGET_VOL` | 0.10 | Moderate risk budget |
| Max leverage | `ETF_MAX_LEVERAGE` | 1.0 | Long-only, no leverage |
| Rebalance cadence | `ETF_REBALANCE_DAYS` | 21 | Monthly; low turnover |
| Top-K holdings | — | 5 | Diversified conviction |
| Trend SMA | — | 200 | Canonical regime gate |

## Tests

```bash
python -m pytest tests/test_etf.py -v
```

All tests use deterministic synthetic data (fixed seed) and run fully offline,
including an explicit **no-look-ahead** check.

## Roadmap to a profit-generating ETF machine

**Mandate:** ETFs only. No options, no single names, no futures.

**Central thesis — why this roadmap can actually print money.**
The v1 diagnostics proved the trap: *no single ETF strategy reaches an
institutional Sharpe*, and our sweep showed that simply scaling one mediocre
sleeve up (more vol target, more leverage) **lowers** Sharpe. The only durable,
non-overfit way to build a money machine from ETFs is the one quoted to death
because it is true — diversification across *uncorrelated return sources* is the
sole free lunch:

> Combine several individually-modest, *low-correlation* ETF return sleeves into
> one risk-balanced, volatility-targeted portfolio, then scale the **whole
> portfolio's** risk (not any single sleeve) to the return target under a hard
> drawdown cap.

If four sleeves each earn Sharpe ≈ 0.5 with pairwise correlation ≈ 0.1, the
combined portfolio Sharpe is ≈ 0.5 × √(4 / (1 + 3·0.1)) ≈ **0.77**, and pushing
correlations toward zero or adding a fifth sleeve takes it past **1.0**. That is
the entire game. Every phase below either *adds an uncorrelated sleeve*,
*lowers correlation between sleeves*, or *converts validated Sharpe into CAGR
safely*. Nothing ships without out-of-sample proof.

Profit is the **expected output of this process**, not a promise — each phase is
gated, and a phase that fails its gate is rejected or reworked, never forced.

---

### Phase 0 — Anti-overfitting backbone (build BEFORE any new alpha) ✅ DONE
*Mode: Production Guardian. Without this, every later number is untrustworthy.*

- `etf/validation.py`: **purged, embargoed walk-forward** + **Combinatorial
  Purged Cross-Validation (CPCV)** harness over the existing backtester.
- **Deflated Sharpe Ratio** and **Probability of Backtest Overfitting (PBO)**
  so we can tell real edge from multiple-testing noise.
- Realistic **cost & capacity model**: per-ETF spread + impact as a function of
  traded ADV%, not a flat bps.
- **Gate:** harness reproduces the v1 full-period backtest within tolerance;
  PBO of the current config is measured and reported.

**Phase 0 results (real data, 2007-01-03 → 2026-06-10), run via
`python -m etf.main --mode validate`:**

| Test | Result | Read |
|------|--------|------|
| Deflated Sharpe Ratio | **0.990** | Edge is real, not multiple-testing luck (>0.95 bar) |
| CPCV OOS Sharpe | median 0.63, **P(SR>0)=100%** | Robustly positive across all OOS folds |
| Walk-forward | **80% of folds** positive Sharpe | Time-stable (only 2020-23 fold ~flat) |
| PBO | **52.8%** | ⚠️ Selecting a config by backtest rank does **not** generalize |
| Capacity (<1%/yr drag) | effectively unlimited | ETF liquidity ⇒ no capacity constraint |

**Decision (the key Phase 0 takeaway):** the single-sleeve trend/momentum edge
is statistically real but weak (Sharpe ≈ 0.4 net / 0.6 gross), and **PBO > 50%
proves parameter-tuning this sleeve is overfitting**. Therefore we **freeze the
robust parameter-free defaults** and pursue Sharpe uplift exclusively by adding
*uncorrelated sleeves* (Phase 2) — which is precisely the program thesis.

### Phase 1 — Harden Sleeve A (Trend / Time-Series Momentum)
*The crisis-armor sleeve we already have, made sharper.*

- True **GEM-style dual momentum**: explicit relative-momentum equity↔bond
  switch, plus accelerating / multi-horizon momentum blend.
- **Gate:** OOS (walk-forward) Sharpe uplift vs v1 with stable fold-by-fold
  behavior; no degradation of the -18% drawdown profile.

### Phase 2 — Add UNCORRELATED sleeves (the real Sharpe engine)
*Each sleeve must be individually OOS-positive and pairwise corr ≲ 0.3.*

- **Sleeve B — Short-horizon mean reversion / dip-buying.** ✅ BUILT & VALIDATED.
  Connors-style RSI(2) oversold entries on broad equity ETFs *only while above
  their 200-day trend*. ETF-only, fully spot, daily cadence.
- **Sleeve C — Defensive carry / duration & gold.** ✅ BUILT & VALIDATED.
  Antonacci-style absolute (time-series) momentum on an **equity-free** universe
  (TLT/IEF/LQD/GLD): hold a defensive asset only while it is trending up *and*
  has positive skip-month momentum. Structurally non-equity-beta — pays in the
  equity-momentum dead zones / flights-to-safety.
- **Sleeve D — Cross-sectional relative strength.** ❌ BUILT & REJECTED.
  Dollar-neutral long/short risk-adjusted momentum across ~10 equity-sector/
  region ETFs. **Most orthogonal source built (corr 0.06–0.18) but NO standalone
  edge:** gross-of-cost Sharpe ≈ 0/negative across an 8-cell parameter sweep
  (−0.31…+0.03), even at zero cost. Sector momentum among so few broad ETFs is
  empirically absent. Adding it via inverse-vol *dragged* the blend (0.51→0.40),
  so it is excluded from the production roster. Class retained as tested
  infrastructure for future research (much wider cross-section) — see
  `CrossSectionalSleeve`.
- **Gate per sleeve:** positive OOS Sharpe standalone; pairwise correlation to
  existing sleeves measured and ≲ 0.3; survives cost model.

**Phase 2 results (real data, 2007 → 2026), via
`python -m etf.main --mode sleeves --start 2007-01-01`:**

| metric | A (trend) | B (mean-rev) | C (defensive) | inverse-vol blend |
|---|--:|--:|--:|--:|
| CAGR | 7.66% | 4.57% | 5.06% | 5.83% |
| Annual vol | 10.81% | 7.78% | 8.16% | **5.49%** |
| Sharpe | 0.46 | 0.23 | 0.28 | **0.51** |
| Sortino | 0.57 | 0.19 | 0.34 | **0.61** |
| Max drawdown | −19.68% | −17.54% | −14.22% | **−10.71%** |
| Calmar | 0.39 | 0.26 | 0.36 | **0.54** |

Pairwise daily-return correlation:

| | A | B | C |
|---|--:|--:|--:|
| A trend | 1.000 | 0.418 | 0.243 |
| B mean-rev | 0.418 | 1.000 | **−0.008** |
| C defensive | 0.243 | −0.008 | 1.000 |

OOS robustness (CPCV): A median 0.68 / P(SR>0) 100%; B median 0.53 / 100%;
C median 0.63 / 92.9%. Sleeve-C parameter sweep (lookback×SMA, 18 cells):
**every cell positive Sharpe** (0.14–0.40), default sits mid-pack → not overfit.

**Decision: SLEEVE C PROMOTED (the thesis works).** Sleeve C is the structural
diversifier Sleeve B couldn't be: it is **uncorrelated to mean reversion
(−0.008)** and low to trend (0.243). The parameter-free inverse-vol blend of all
three sleeves now **lifts Sharpe to 0.51 (> best single 0.46)** — a genuine
risk-adjusted improvement, not just drawdown reduction — while **cutting MaxDD
46%** (−19.7%→−10.7%), **halving vol** (10.8%→5.5%), and **lifting Calmar 38%**
(0.39→0.54). All three sleeves are OOS-positive.

*Honest gap:* blended Sharpe 0.51 is still below the Phase 3 money gate (≥1.10).
Two levers close it, both already designed: (1) **Sleeve D** (cross-sectional
relative strength) adds a fourth low-correlation source; (2) the **Phase 3
risk-parity combiner** (a richer version of this inverse-vol proxy) extracts
more diversification than equal-risk-naive blending. The 5.5% blended vol also
leaves large headroom under the 10% budget for Phase 4's capped leverage to lift
CAGR without breaching risk limits.

### Phase 3 — Portfolio construction across sleeves
*Where modest sleeves become an institutional-grade book.*

- **Equal-risk-contribution (risk-parity) allocation across sleeves**, then
  portfolio-level volatility targeting (reuse the existing vol-target machinery).
- Correlation-aware sizing so the book auto-tilts toward whatever is currently
  diversifying.
- **Gate (this is the money-machine bar):** combined **OOS Sharpe ≥ 1.10,
  MaxDD ≤ 18%, Calmar ≥ 0.80, Profit factor ≥ 1.20** — the Research→Paper gate.

**Phase 3 results (real data, 2007 → 2026), via
`python -m etf.main --mode portfolio --start 2007-01-01`** — combiner allocates
*capital across the three sleeve return streams* with a causal trailing
covariance (126-day window, lagged 1 day, rebalanced every 21 days), then scales
the book to a 10% vol target (leverage capped at 1.0×):

| method | Sharpe | Sortino | CAGR | Vol | MaxDD | Calmar | PF |
|---|--:|--:|--:|--:|--:|--:|--:|
| equal | **0.46** | 0.57 | 5.84% | 6.27% | −11.33% | **0.52** | 1.19 |
| inverse-vol | 0.44 | 0.53 | 5.58% | 5.86% | −12.27% | 0.45 | 1.21 |
| ERC (risk-parity) | 0.42 | 0.50 | 5.44% | 5.82% | −12.51% | 0.43 | **1.20** |

OOS robustness of the ERC book (CPCV 8C2, purge 5): **median Sharpe 0.96,
P(SR>0) = 100%, DSR = 0.918**. Average capital allocation when deployed: trend
21%, mean-rev 37%, defensive 41% (avg gross 97%).

**Decision: GATE NOT CLEARED — reported honestly.** Three findings, all
empirically grounded:

1. **ERC did *not* beat naive equal/inverse-vol here.** With pairwise
   correlations already low (−0.01 to 0.42), there is little covariance
   structure for risk-parity to exploit; ERC's only material effect is to
   *under-weight the highest-Sharpe sleeve* (trend, 0.46) in favour of the
   lower-Sharpe defensives — which mechanically drags realized Sharpe. This is
   the textbook risk-parity failure mode: it optimises risk balance, not
   return, and is only superior when a high-risk sleeve is *also* the
   low-Sharpe one. Here it is the opposite. **Equal-weight is retained as the
   honest default; ERC is kept as a selectable method, not promoted.**
2. **The covariance-based combiner (0.42–0.46) slightly trails the simpler
   63-day inverse-vol proxy (0.51)** from Phase 2. The 126-day covariance window
   is less responsive than the 63-day vol proxy, and the combiner additionally
   charges its own rebalancing turnover. Conclusion: *for three already-
   uncorrelated sleeves, sophistication did not add value over simple blending.*
3. **The OOS signal is genuinely strong (DSR 0.918, P(SR>0) 100%)** even though
   full-period realized Sharpe (0.42–0.46) is below the 1.10 gate. The book is
   *real and robust* — it is simply *under-powered*. The gap is a **return**
   gap, not a validity gap.

*Path forward (honest):* clearing the 1.10 Sharpe gate from three modest,
genuinely-uncorrelated sleeves is not achievable by combiner sophistication
alone — it requires **more orthogonal edge** (additional validated sleeves) or
acceptance that the *risk-adjusted* product is ~0.5 Sharpe and the **CAGR** lever
(Phase 4 capped leverage, well within the 6% realized vs 10% budget) is what
turns it into a competitive return stream. We do **not** overfit knobs to fake a
1.10. The combiner, ERC solver, vol-target, and method comparison are all built,
tested (7 dedicated tests, no look-ahead verified), and production-shaped.

### Phase 4 — Convert Sharpe into CAGR safely (the return lever)
*High Sharpe + low return = under-levered. This phase fixes CAGR — carefully.*

- Apply **rules-based, capped leverage via IBKR Reg-T margin on liquid ETFs**
  so *realized portfolio vol matches the risk budget* — leverage serves the
  risk target, it does not chase return beyond it.
- **Dynamic de-leveraging** on vol spikes and drawdowns (extend the existing
  drawdown overlay to act on gross exposure).
- Evaluate (and most likely **reject**) leveraged ETFs: document daily-reset
  decay/path-dependency; only admissible as bounded tactical tools, never core.
- **Gate:** levered MaxDD still ≤ target in a full crisis replay (2008/2020/
  2022); no path-dependent blow-up; cost-stress (×2 slippage) still profitable.

**Phase 4 results (real data, 2007 → 2026), via
`python -m etf.main --mode portfolio --max-leverage <cap> --derisk`** — combiner
gross capped at `cap`, margin charged at rf + 150 bps on the levered portion, a
book-level drawdown circuit-breaker (`--derisk`) that scales gross down during
equity drawdowns. Leverage sweep (equal-weight combiner):

| cap | derisk | Sharpe | CAGR | Vol | MaxDD | Calmar | avg gross |
|---|---|--:|--:|--:|--:|--:|--:|
| 1.00 | off | **0.46** | 5.85% | 6.27% | −11.33% | **0.52** | 97% |
| 1.25 | ON | 0.39 | 5.85% | 7.70% | −13.18% | 0.44 | 119% |
| 1.50 | ON | 0.34 | 5.81% | 8.86% | −14.50% | 0.40 | 137% |
| 2.00 | off | 0.33 | 6.04% | 10.44% | −18.29% | 0.33 | 163% |
| 2.00 | ON | 0.27 | 5.27% | 9.79% | −16.92% | 0.31 | 152% |

**Decision: LEVERAGE REJECTED AS A RETURN LEVER — reported honestly.** The sweep
is unambiguous: pushing gross from 1.0× to 2.0× *barely* moves CAGR (5.85% →
6.04%, +0.19 pp) while **degrading Sharpe 0.46 → 0.33** and **deepening MaxDD
−11% → −18%**. Leverage cannot manufacture Sharpe; it only scales an existing
~0.46-Sharpe stream up *and* adds a margin-cost drag, so risk-adjusted quality
strictly falls. The earlier "6% realized vs 10% budget ⇒ headroom for leverage"
reasoning was **wrong**: the unlevered book's realised vol already reflects the
combiner's risk balance, and adding gross raised vol faster than return.

Two findings that *do* survive and are kept:

1. **The drawdown circuit-breaker works as designed and is free insurance.** In
   crisis replay (cap 2.0×) it cut the **2022 bear** window drawdown from −12.0%
   to **−8.6%** and **COVID-2020** from −18.1% to **−16.7%**, at negligible CAGR
   cost. It is retained (recommend `dd_derisk=ON` whenever gross > 1.0×).
2. **The book is cost-robust.** At cap 1.5× with **×2 slippage** it stays
   profitable (PF 1.12, Sharpe 0.26, MaxDD unchanged at −14.6%) — no fragility.

*Honest conclusion:* the production-recommended config is **cap = 1.0× (no
leverage) with the circuit-breaker armed** — leverage is built, tested (5
dedicated tests incl. levered no-look-ahead), and *available*, but the evidence
says it is the **wrong lever for this book**. The gate (Sharpe ≥ 1.10) is a
**Sharpe** problem, and Sharpe comes only from **more orthogonal edge** (a 4th
genuinely-uncorrelated sleeve) — not from leverage and not from combiner
sophistication. We do not lever into a mediocre Sharpe to fake CAGR.

### Phase 5 — Execution realism & live readiness
*Mode: Production Guardian.*

- Marketable-limit execution with spread/impact awareness; fractional shares;
  rebalance-threshold tuning to minimize turnover drag.
- Kill-switch + reconciliation integration; restart/state-recovery safety.
- **Gate (Paper→Live):** paper ≥ 20 trading days; live-vs-modeled slippage
  deviation ≤ 20%; order rejection ≤ 1%; zero unresolved reconciliation
  mismatches; documented runbook + rollback.

**Phase 5 progress — backtest/live unification (the critical fix).** The live
`signal`, `paper`, and `live` paths previously traded the *single* trend
strategy (`compute_target_weights`) while the *validated* book is the
three-sleeve combiner — a silent backtest/live divergence that violates the
"keep backtest, paper, and live logic aligned" rule. Fixed:

- New `etf.portfolio.live_target_weights(prices, cfg, sleeves)` produces **today's
  combined ETF book** with the *exact* allocation logic of
  `run_combined_backtest`: each sleeve's current target weights, combined by the
  trailing-covariance cross-sleeve allocation (method/vol-target/leverage-cap/
  circuit-breaker all honoured). The combined weight per symbol is provably
  `Σ_i alloc_i · sleeve_w_i[sym]` (regression-tested).
- `--mode signal` now prints the combined book, sleeve capital allocation, gross
  and cash; `--mode paper/live` trade that same book through the existing
  fail-safe IBKR broker (dry-run default, churn threshold, readonly block).
- 5 dedicated consistency tests: weight reconstruction, gross ≤ leverage cap,
  determinism + causality under truncation, equal-method balance, and live
  de-risk response. **Full ETF suite green.**

Verified live signal (2026-06-12, equal combiner): combined book holds **LQD
33%** (defensive carry) + equity names (IWM/QQQ/XLE/XLK/EEM) from the trend &
mean-rev sleeves, **48% cash** — exactly the diversified, risk-balanced book the
backtest validated.

**Phase 5 progress — marketable-limit execution + post-trade reconciliation.**
The broker previously submitted *market orders only*, ignoring the
`execution.order_type="LMT"`/`limit_offset_bps` config — meaning no protection
against a blown-out spread or a stale quote. Fixed:

- `marketable_limit_price(action, ref, offset_bps)` (pure, penny-rounded): BUY
  crosses up `ref·(1+offset)`, SELL crosses down `ref·(1−offset)`. A marketable
  limit fills like a market order in normal liquidity yet **caps slippage** if
  the book is thin. `plan_rebalance` attaches `order_type`/`limit_price` to each
  `PlannedOrder`; `execute_orders` submits a `LimitOrder` when `order_type==LMT`,
  else `MarketOrder` — both still behind the readonly/dry-run guards.
- `compute_reconciliation(target, positions, prices, equity, tol)` (pure) and
  `IBKRETFBroker.reconcile(target, cfg)` (fail-safe): after every paper/live
  rebalance, realised broker weights are compared to the target book and any
  `|Δw| > min_rebalance_delta` is flagged for the runbook — directly serving the
  Paper→Live gate's *zero unresolved reconciliation mismatches*. Wired into
  `_trade`, so `paper`/`live` now log an explicit OK/MISMATCH line each cycle.
- 12 dedicated unit tests (`tests/test_etf_broker.py`): limit-side crossing,
  penny rounding, zero/negative-offset clamps, non-positive-price guard, and
  five reconciliation cases (match, under-fill drift, unexpected position,
  zero-equity fail-safe, missing-price). **Full ETF suite green (78 tests).**

**Phase 5 progress — kill-switch + slippage telemetry (`etf/safety.py`).** The
combined book now has an ETF-native safety layer (self-contained — no dependency
on the equities `core.kill_switch`):

- **Pre-trade kill-switch** `pretrade_safety_check(cfg, ...)` runs before every
  rebalance. It HALTS (kill-switch, human reset required) on catastrophic
  conditions — book drawdown ≥ `risk.hard_halt_drawdown` (25%) or single-day
  loss ≤ −`risk.max_daily_loss` (8%) — and BLOCKS (skip this cycle, retry next)
  on recoverable problems: stale/missing data, an unresolved reconciliation
  mismatch, or gross exposure over the leverage cap. Distinct from the smooth
  `dd_derisk` overlay: the overlay gently scales down, the kill-switch slams the
  brakes. Wired into `_trade` — `paper`/`live` abort with a logged reason before
  any order is submitted.
- **Slippage telemetry** `compute_slippage(orders, fills, cfg)` (pure) records
  signed adverse slippage (bps + dollars) per fill vs the plan's assumed price,
  and flags when realised average exceeds the 20%-over-budget gate
  (`execution.slippage_bps × 1.2`). `IBKRETFBroker.collect_fills` captures
  realised `avgFillPrice` post-execution; `log_slippage` appends one JSONL row
  per cycle to `execution.slippage_log` (fail-safe — telemetry can never crash
  the trading loop).
- 19 dedicated unit tests (`tests/test_etf_safety.py`): all kill-switch triggers
  + boundary cases, slippage sign convention (adverse vs improvement), aggregate
  weighting, unfilled-order skipping, within/over tolerance, and JSONL
  append/fail-safe. **Full ETF suite green (97 tests).**

**Phase 5 progress — persistent equity-state tracking (`etf/state.py`).** The
kill-switch needs live drawdown / daily-P&L, which require *memory across cycles*
that survives restarts. `update_state(prev, equity)` (pure) maintains the running
peak (high-water mark) and the start-of-day baseline, returning
`current_drawdown` and `daily_pnl_pct` for the kill-switch; `save_state`/
`load_state` persist a tiny JSON atomically (`os.replace`, so a crash mid-write
can't corrupt it) at `execution.state_path`. Wired into `_trade`: each cycle
reads the broker equity, advances the state, and feeds *real* drawdown/daily-P&L
into `pretrade_safety_check`. 11 unit tests (`tests/test_etf_state.py`) incl. a
restart-simulation that proves the peak survives a process bounce so a −15%
drawdown is still detected after "restart". The kill-switch's catastrophic
triggers are now fully live (no more 0.0 placeholders).

*Still open in Phase 5:* the ≥ 20-day paper run that the Paper→Live gate requires
(now unblocked — all live-safety rails are in place).


### Phase 5b — 4th-sleeve research (honest rejections)

The Phase 3 ERC combiner reached an OOS median Sharpe of 0.96 but the
Research→Paper gate wants ≥ 1.10. Two orthogonal-sleeve candidates were built and
evaluated against the full 2007–2026 sample under realistic costs — **both
rejected**, and the negative result is itself the finding:

| candidate | corr to trend | standalone Sharpe (gross-rf) | CPCV median | DSR | blend impact | verdict |
|---|---|---|---|---|---|---|
| Cross-sectional L/S (D) | 0.06–0.18 | ~0 / negative | — | — | 0.51 → 0.40 | **reject** |
| Turn-of-month seasonality (E) | 0.365 | −0.04 | 0.31 | 0.020 | 0.51 → 0.39 | **reject** |

Seasonality is the more instructive failure: the turn-of-month premium on three
broad equity ETFs, net of daily-rebalanced transaction costs, has **no standalone
edge** (DSR 0.020 — nothing survives multiple-testing deflation) and is the
*least* orthogonal candidate (0.365 to trend — it is simply long equity beta
inside a calendar window). Adding it dragged the parameter-free inv-vol blend
0.51 → 0.39.

**Lesson:** the Sharpe gate is an **edge problem, not an orthogonality problem.**
The three production sleeves are already decently uncorrelated; bolting on
another low/zero-edge orthogonal stream only adds noise. Both classes are kept as
tested research infrastructure (`CrossSectionalSleeve`, `TurnOfMonthSleeve`,
`_tom_in_window`) — evaluable any time via `--mode sleeves --candidate {seasonality,cross_sectional}`
— but the production roster stays at three. The next experiments must target
genuinely new *edge* (e.g. volatility-managed equity, duration carry-momentum),
not more diversification.


### Phase 6 — Continuous governance (keep it printing)
- Experiment log (hypothesis, code hash, config, dates, metrics, decision).
- Live **regime monitoring & drift detection**; scheduled re-validation.
- **Live scale-up gate:** start ≤ 25% of target capital; expand only after two
  clean review cycles with live stability within 10% of paper baselines.

---

### Definition of done (the machine)
A multi-sleeve, ETF-only portfolio that, **out-of-sample and under realistic
costs**, clears Sharpe ≥ 1.10 / MaxDD ≤ 18% / Calmar ≥ 0.80, is risk-scaled to a
target CAGR that beats buy-and-hold SPY on a risk-adjusted basis, and runs live
on IBKR with kill-switch, reconciliation, and drift monitoring. Profit is the
designed consequence of stacking validated, uncorrelated edges — not a forecast.
