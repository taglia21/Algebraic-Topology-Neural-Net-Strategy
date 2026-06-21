# Local Pre-Commit Hook Setup

This guide explains how to set up a local Git pre-commit hook to validate promotion gates before commits (optional but recommended).

## What It Does

The pre-commit hook:
- Detects staged promotion gate evidence files
- Automatically runs gate validation
- **Blocks commit** if gate validation fails
- Provides immediate feedback before push

## Installation

### Step 1: Copy Hook to Git Hooks Directory
```bash
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Step 2: Test Installation
```bash
# This should output "hook installed"
ls -lh .git/hooks/pre-commit
```

## Usage

### When Promoting Research → Paper

1. Create evidence:
```bash
cp templates/promotion_gate_evidence.research_to_paper.example.json \
   templates/promotion_gate_evidence.research_to_paper.json
# Edit with your actual backtest evidence
```

2. Stage and commit:
```bash
git add templates/promotion_gate_evidence.research_to_paper.json
git commit -m "promotion: research->paper gate evidence"
```

3. Hook runs automatically:
```
[pre-commit] Checking for promotion gate evidence...
[pre-commit] Promotion evidence detected. Validating...
Validating research->paper gate: templates/promotion_gate_evidence.research_to_paper.json
[pre-commit] research->paper gate PASSED
[pre-commit] All promotion gates passed. Proceeding with commit.
```

### If Validation Fails

```bash
git add templates/promotion_gate_evidence.paper_to_live.json
git commit -m "promotion: paper->live evidence"

# Output:
# [pre-commit] paper->live gate FAILED
# Gate: paper_to_live
# Status: FAIL
# Checks:
# - [X] Paper period >= 20 trading days | actual=15 | expected=ge 20
# ...
# [pre-commit] pre-commit hook failed
# Error: Your commit was blocked because the gate did not pass.
# Fix the evidence values and try again.
```

## Disabling Hook Temporarily

If you need to bypass the hook for a non-promotion commit:

```bash
# Bypass hook for this commit only
git commit --no-verify

# But use this sparingly—the hook exists for safety
```

## Removing Hook

```bash
rm .git/hooks/pre-commit
```

## Advanced: Per-Developer Setup

If multiple developers use this repo, you can share hooks via a shared directory:

```bash
# Store shared hooks in repo
mkdir -p .githooks
cp scripts/pre-commit-hook.sh .githooks/pre-commit
chmod +x .githooks/pre-commit

# Configure git to use them
git config core.hooksPath .githooks

# Other developers do the same
git config core.hooksPath .githooks
```

Then the hook will be shared in version control and automatically used by all developers.

## Troubleshooting

### "Permission denied"
```bash
chmod +x .git/hooks/pre-commit
```

### "Python: command not found"
Ensure Python 3.12+ is in your PATH:
```bash
which python3
python3 --version
```

### "Gate validation script not found"
Ensure you're running from the repository root:
```bash
cd /path/to/Algebraic-Topology-Neural-Net-Strategy
```

---

## See Also
- [CI_AND_GOVERNANCE_SETUP.md](CI_AND_GOVERNANCE_SETUP.md) — GitHub Actions CI workflow
- [PR_WORKFLOW.md](PR_WORKFLOW.md) — Full PR-first workflow
- [../scripts/check_promotion_gates.py](../scripts/check_promotion_gates.py) — Gate validation logic
