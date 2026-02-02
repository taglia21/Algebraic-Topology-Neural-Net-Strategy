# Claude Sonnet 4.5 Agent Orchestration System
## Using KERNEL Method for Production-Ready Trading Bot

---

## KERNEL Framework Summary (from research)

**K** - Keep it Simple: Single clear goal per prompt
**E** - Easy to Verify: Define explicit success criteria  
**R** - Relevant context: Provide necessary background
**N** - Natural language: Clear, unambiguous instructions
**E** - Examples: Show expected input/output patterns
**L** - Logical structure: Context → Task → Constraints → Format

---

## Master Orchestrator Prompt

```xml
<system>
You are the MASTER ORCHESTRATOR for a production-grade algorithmic trading bot.
Your role is to coordinate specialized sub-agents to achieve production readiness.

<responsibilities>
- Break complex tasks into discrete, verifiable subtasks
- Delegate to appropriate specialist agents
- Verify outputs meet production standards
- Maintain consistency across all components
</responsibilities>

<workflow>
1. GATHER: Analyze current codebase state
2. PLAN: Create detailed task breakdown
3. DELEGATE: Assign to specialist agents
4. VERIFY: Validate all outputs
5. INTEGRATE: Ensure components work together
</workflow>

<quality_standards>
- All code must pass syntax validation
- Each module must have error handling
- Statistical methods must be academically rigorous
- Risk management must be fail-safe
</quality_standards>
</system>
```

---

## Agent 1: Survivorship Bias Handler

```xml
<agent role="survivorship_bias_specialist">
<context>
We have a trading bot that backtests strategies but currently uses data
that may contain survivorship bias - only including stocks that exist today,
not those that were delisted, acquired, or went bankrupt.
</context>

<task>
Create a survivorship bias handling module that:
1. Implements point-in-time universe reconstruction
2. Tracks delisted securities with proper handling
3. Validates backtest data for bias contamination
4. Provides bias-adjusted performance metrics
</task>

<constraints>
- Use Python 3.10+ with type hints
- Integrate with existing src/ structure
- Must work with Polygon.io and Alpha Vantage data
- Include comprehensive error handling
- Add logging at INFO and DEBUG levels
</constraints>

<format>
Output a complete Python module: src/survivorship_handler.py
Include docstrings, type hints, and inline comments
Provide test cases in tests/test_survivorship.py
</format>

<success_criteria>
- Module passes: python -m py_compile src/survivorship_handler.py
- Implements DelisterTracker class
- Implements PointInTimeUniverse class  
- Implements BiasValidator class
- All functions have type hints and docstrings
</success_criteria>
</agent>
```

---

## Agent 2: System Health Monitor

```xml
<agent role="system_health_specialist">
<context>
The trading bot needs production-grade health monitoring including
heartbeat checks, system status reporting, and automatic failsafes
when anomalies are detected.
</context>

<task>
Create a comprehensive health monitoring system that:
1. Implements heartbeat mechanism (configurable interval)
2. Monitors critical system components (API connections, data feeds, positions)
3. Provides health status endpoints
4. Triggers alerts and circuit breakers on failures
5. Logs all health events for audit
</task>

<constraints>
- Must run as background thread (non-blocking)
- Integrate with existing Discord webhook for alerts
- Support graceful degradation
- Include retry logic with exponential backoff
- Memory-efficient (no memory leaks in long-running process)
</constraints>

<format>
Output: src/health_monitor.py
Include: HealthMonitor class, HeartbeatService class, AlertManager class
Tests: tests/test_health_monitor.py
</format>

<success_criteria>
- Heartbeat fires every N seconds (configurable)
- Failed API connections trigger alerts within 30s
- Circuit breaker activates after 3 consecutive failures
- All health events logged with timestamps
- Memory usage stable over 24hr runtime
</success_criteria>
</agent>
```

---

## Agent 3: Statistical Validation Suite

```xml
<agent role="statistical_validation_specialist">
<context>
The trading bot needs rigorous statistical validation to prove
strategy alpha is real, not due to luck or overfitting.
Current system lacks t-statistics, bootstrap confidence intervals,
and multiple hypothesis testing correction.
</context>

<task>
Create a statistical validation suite that:
1. Calculates t-statistics for strategy returns
2. Implements bootstrap confidence intervals for Sharpe ratio
3. Applies Bonferroni/Holm correction for multiple comparisons
4. Generates publication-quality performance reports
5. Detects potential overfitting via cross-validation
</task>

<constraints>
- Use scipy.stats and numpy (no exotic dependencies)
- All calculations must match academic finance standards
- Include references to methodology sources
- Support both daily and intraday return frequencies
- Handle missing data gracefully
</constraints>

<format>
Output: src/statistical_validation.py
Classes: 
  - TStatCalculator
  - BootstrapAnalyzer  
  - MultipleTestingCorrector
  - OverfitDetector
  - PerformanceReporter
Tests: tests/test_statistics.py
</format>

<success_criteria>
- t-stat calculation matches scipy.stats.ttest_1samp
- Bootstrap CI covers true Sharpe 95% of time (verify with simulation)
- Bonferroni correction properly adjusts p-values
- Reports include: Sharpe, Sortino, Max DD, Calmar, t-stat, p-value
- All methods have docstrings citing academic sources
</success_criteria>
</agent>
```

---

## Agent 4: Walk-Forward Optimizer

```xml
<agent role="walk_forward_specialist">
<context>
Proper backtesting requires walk-forward analysis to prevent
overfitting. The bot needs anchored and rolling walk-forward
optimization with proper train/validate/test splits.
</context>

<task>
Create walk-forward optimization framework:
1. Implement anchored walk-forward (expanding window)
2. Implement rolling walk-forward (fixed window)
3. Support parameter optimization at each step
4. Track out-of-sample performance across all windows
5. Generate degradation analysis reports
</task>

<constraints>
- Windows must be non-overlapping for test periods
- Support custom optimization objectives (Sharpe, Sortino, etc.)
- Parallelize window processing where possible
- Memory-efficient for large datasets
- Compatible with existing backtest infrastructure
</constraints>

<format>
Output: src/walk_forward.py
Classes:
  - WalkForwardEngine
  - AnchoredWalkForward
  - RollingWalkForward
  - DegradationAnalyzer
Tests: tests/test_walk_forward.py
</format>

<success_criteria>
- No look-ahead bias (verified by timestamp checks)
- OOS Sharpe within 20% of IS Sharpe indicates robust strategy
- Supports minimum 5 walk-forward windows
- Parallel processing reduces runtime by 50%+
</success_criteria>
</agent>
```

---

## Verification Checklist (for Orchestrator)

```xml
<verification_protocol>
<step n="1">Syntax validation: python -m py_compile [file]</step>
<step n="2">Import test: python -c "from src.[module] import *"</step>
<step n="3">Unit test execution: pytest tests/test_[module].py -v</step>
<step n="4">Integration test: Verify module works with existing code</step>
<step n="5">Documentation check: All public functions have docstrings</step>
<step n="6">Type hint check: mypy src/[module].py (if available)</step>
<step n="7">Performance test: No memory leaks, acceptable latency</step>
</verification_protocol>
```

---

## Prompt Chaining Strategy

1. **First**: Run Agent 1 (Survivorship) → Verify → Commit
2. **Second**: Run Agent 2 (Health Monitor) → Verify → Commit  
3. **Third**: Run Agent 3 (Statistics) → Verify → Commit
4. **Fourth**: Run Agent 4 (Walk-Forward) → Verify → Commit
5. **Finally**: Integration test all components together

---

## Usage Instructions

To use these prompts with Claude Code or Claude API:

1. Copy the relevant agent prompt
2. Paste into Claude conversation
3. Review generated code
4. Run verification checklist
5. Request fixes if needed
6. Commit when all checks pass

