# Branch Protection Setup (Manual)

Since the automated script requires admin role, here's the manual setup:

## GitHub Web UI Method

1. Go to: https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy/settings/branches

2. Click "Add rule" under "Branch protection rules"

3. Configure:
   - **Branch name pattern:** `main`
   - ✅ Require a pull request before merging
     - ✅ Require approvals: 1
     - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require status checks to pass before merging
     - ✅ Require branches to be up to date before merging
   - ✅ Restrict who can push to matching branches
   - ✅ Include administrators

4. Click "Create" (or "Save changes" if updating existing rule)

## Verification

After setup, the following will be enforced:

```
Direct push to main
  ↓
BLOCKED ✗ → "branch protection rules"

PR without approval
  ↓
BLOCKED ✗ → "requires approving reviews"

PR with failing CI (gates, tests, etc.)
  ↓
BLOCKED ✗ → "status checks failed"

PR with approval + passing CI
  ↓
ALLOWED ✓ → Can merge
```

## Automation Script (if credentials available)

If you add GitHub credentials later, the script can apply settings programmatically:

```bash
bash scripts/enable_branch_protection.sh
```

See: [scripts/enable_branch_protection.sh](scripts/enable_branch_protection.sh)
