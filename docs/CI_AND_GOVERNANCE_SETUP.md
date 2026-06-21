# CI & Governance Setup Guide

This document explains the automated governance infrastructure added to enforce promotion gates and prevent direct-to-main pushes.

## What Was Added

### 1. GitHub Actions Workflow: Promotion Gates Validation
**File:** `.github/workflows/promotion-gates.yml`

**Triggers on:**
- Any PR to `main` that modifies promotion evidence JSON files
- Changes to the gate checker script or workflow itself

**Behavior:**
- Detects which gates are being validated (research→paper, paper→live, live scale-up)
- Runs `scripts/check_promotion_gates.py` for each detected gate
- Fails the workflow if any gate doesn't pass
- Prevents merge until gates pass

**Example:**
When you open a PR with `templates/promotion_gate_evidence.paper_to_live.json` added or modified:
```
✓ Checkout code
✓ Set up Python 3.12
✓ Detect promotion evidence files
  Found: promotion_gate_evidence.paper_to_live.json
✓ Validate paper->live gate
  Result: PASS
✓ Report results
```

---

### 2. Branch Protection Configuration Script
**File:** `scripts/enable_branch_protection.sh`

**What it does:**
- Enables GitHub's built-in branch protection on `main`
- Configures via GitHub API (idempotent)
- Requires PR before merge
- Requires 1 approving review
- Dismisses stale reviews on new commits
- Enforces for administrators

**Usage:**
```bash
bash scripts/enable_branch_protection.sh
```

Or manually apply settings in GitHub web UI:
Settings → Branches → Branch protection rules → Add rule for `main`

---

## How to Use

### Phase 1: Enable Automation (Admin Only)

1. **Enable CI workflow:**
   - Already in repository; will activate on next PR with promotion evidence

2. **Enable branch protection:**
   ```bash
   bash scripts/enable_branch_protection.sh
   ```
   - Requires `gh` CLI and admin/maintain role
   - Or apply manually via GitHub web UI (Settings → Branches)

### Phase 2: Use PR Workflow (Everyone)

For any code change:

```bash
# 1. Create feature branch from main
git checkout main && git pull
git checkout -b feat/my-feature

# 2. Make changes and test locally
python -m pytest tests/ -v

# 3. Commit and push
git add -A
git commit -m "feat: description"
git push -u origin feat/my-feature

# 4. Open PR
gh pr create --base main --head feat/my-feature --title "..." --body "..."

# 5. Wait for CI + review, then merge
```

Branch protection will prevent merge if:
- CI (including gate checks) hasn't passed
- No approvals exist
- Outstanding conversations

### Phase 3: Promotion Gate Validation (Paper→Live)

When ready to promote from paper to live:

1. **Collect evidence** during 20+ days of paper trading:
   - Trading days elapsed
   - Realized vs. modeled slippage (bps)
   - Order rejection rate
   - Unresolved reconciliation mismatches
   - Kill-switch halts due to defects
   - Runbook/rollback documented

2. **Create evidence JSON:**
   ```json
   {
     "paper_trading_days": 24,
     "modeled_slippage_bps": 7.0,
     "realized_slippage_bps": 8.1,
     "order_rejection_rate": 0.003,
     "unresolved_reconciliation_mismatches": 0,
     "kill_switch_halts_due_to_software_defect": 0,
     "runbook_and_rollback_documented": true
   }
   ```

3. **Open PR with evidence:**
   ```bash
   cp templates/promotion_gate_evidence.paper_to_live.example.json \
      templates/promotion_gate_evidence.paper_to_live.json
   # Edit with your actual evidence values
   git add templates/promotion_gate_evidence.paper_to_live.json
   git commit -m "promotion: submit paper->live gate evidence"
   git push -u origin feat/paper-to-live-promotion
   gh pr create --base main ...
   ```

4. **CI automatically validates:**
   - GitHub Actions runs gate checker
   - Reports pass/fail in workflow
   - PR merge is blocked if gate fails
   - PR can merge only if gate passes + approval given

---

## Architecture Diagram

```
Local Development
    ↓
git checkout -b feat/...
    ↓
Make changes + commit
    ↓
git push origin feat/...
    ↓
gh pr create (opens PR to main)
    ↓
GitHub Branch Protection
├─ Requires 1 approving review
├─ Requires CI to pass
└─ If promotion evidence:
       ↓
       GitHub Actions Workflow
       ├─ Detects gate type
       ├─ Runs check_promotion_gates.py
       └─ Pass/Fail gates
            ↓ (if fail)
            CI red → PR merge blocked
            ↓ (if pass)
            CI green → PR can merge with approval
    ↓
Code Review + Approval
    ↓
Merge to main (automatic or manual)
```

---

## Testing the Setup

### Test 1: Verify CI workflow syntax
```bash
cd .github/workflows
python -m json.tool promotion-gates.yml  # Check YAML validity
```

### Test 2: Manual gate validation
```bash
python scripts/check_promotion_gates.py \
  --gate paper_to_live \
  --input templates/promotion_gate_evidence.paper_to_live.example.json \
  --report-out /tmp/test_report.json

# Should output: Status: PASS, exit code 0
```

### Test 3: Branch protection (requires admin)
```bash
bash scripts/enable_branch_protection.sh
# Then verify in: https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy/settings/branches
```

---

## Troubleshooting

### "CI workflow not triggering"
- Verify `.github/workflows/promotion-gates.yml` exists
- Check that PR modifies a file matching `templates/promotion_gate_evidence*.json`
- Wait ~1 min for GitHub to detect the workflow

### "Gate validation failed but I believe my evidence is correct"
- Run manually: `python scripts/check_promotion_gates.py --gate <gate_type> --input <file>`
- Check gate thresholds in `scripts/check_promotion_gates.py` (search `GateRule`)
- Adjust evidence or gate thresholds, re-test

### "Branch protection script fails with 'Cannot access repository'"
- Ensure `gh` CLI is authenticated: `gh auth status`
- Verify user has admin/maintain role on repo
- Try: `gh repo view taglia21/Algebraic-Topology-Neural-Net-Strategy`

### "How do I bypass branch protection in emergency?"
- GitHub allows force-push for admins if protection configured with enforce_admins=false
- Better: use GitHub web UI to temporarily disable protection, merge, re-enable
- Best: follow the PR workflow, always

---

## Maintenance

- **Update gate thresholds:** Edit `scripts/check_promotion_gates.py` GateRule definitions
- **Add new gate types:** Add entries to `_build_rules()` function
- **Customize workflow triggers:** Edit `.github/workflows/promotion-gates.yml` `on:` section
- **Change branch protection rules:** Re-run `scripts/enable_branch_protection.sh` or use GitHub UI

---

## References

- [GitHub Branch Protection Docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [gh CLI Docs](https://cli.github.com/manual/)
- Local setup guide: [docs/PR_WORKFLOW.md](../docs/PR_WORKFLOW.md)
