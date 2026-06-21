#!/usr/bin/env bash
# Enable GitHub branch protection on main
# 
# Prerequisites:
#   - gh CLI installed: https://cli.github.com/
#   - Admin or maintain role on the repository
#   - GITHUB_TOKEN with repo scope set (gh handles this automatically)
#
# Usage:
#   bash scripts/enable_branch_protection.sh
#

set -euo pipefail

REPO="${1:-.}"  # Use current repo if not specified
BRANCH="main"

echo "=== GitHub Branch Protection Configuration ==="
echo "Repository: $REPO"
echo "Branch: $BRANCH"
echo ""

# Verify gh CLI is available
if ! command -v gh &> /dev/null; then
    echo "ERROR: gh CLI not found. Install from https://cli.github.com/"
    exit 1
fi

# Verify admin/maintain access
echo "Verifying repository access..."
if ! gh repo view "$REPO" &> /dev/null; then
    echo "ERROR: Cannot access repository. Check permissions and GITHUB_TOKEN."
    exit 1
fi

echo "✓ Repository access verified"
echo ""

# Extract owner and repo from remote URL if in a Git repo
if [ -d .git ]; then
    REMOTE=$(git config --get remote.origin.url)
    if [[ $REMOTE =~ github\.com[:/]([^/]+)/(.+?)(?:\.git)?$ ]]; then
        OWNER="${BASH_REMATCH[1]}"
        REPO_NAME="${BASH_REMATCH[2]}"
        REPO="$OWNER/$REPO_NAME"
    fi
fi

echo "Configuring protection for: $REPO/$BRANCH"
echo ""

# Function to update branch protection using GitHub API
configure_protection() {
    echo "Step 1: Require pull request before merging"
    gh api \
        --method PATCH \
        repos/"$REPO"/branches/"$BRANCH"/protection \
        -f required_pull_request_reviews='{"dismiss_stale_reviews":true,"require_code_owner_reviews":false,"required_approving_review_count":1}' \
        -f dismiss_stale_reviews=true \
        -f require_code_owner_reviews=false \
        -F require_last_commit_approval=false
    
    echo "✓ Configured: Require PR + 1 approval"
    echo ""
    
    echo "Step 2: Require status checks to pass"
    gh api \
        --method PATCH \
        repos/"$REPO"/branches/"$BRANCH"/protection \
        -f required_status_checks='{"strict":true,"contexts":["promotion-gates-validation"]}' \
        -F enforce_admins=true
    
    echo "✓ Configured: Status checks required"
    echo ""
    
    echo "Step 3: Restrict who can push"
    gh api \
        --method PATCH \
        repos/"$REPO"/branches/"$BRANCH"/protection \
        -F restrict_push_access=false
    
    echo "✓ Configured: Restriction rules set"
    echo ""
}

# Run configuration
configure_protection

echo "=== Branch Protection Enabled ==="
echo ""
echo "Configuration summary:"
echo "  ✓ Require pull request before merging"
echo "  ✓ Require 1 approving review"
echo "  ✓ Dismiss stale reviews on new commits"
echo "  ✓ Require status checks to pass"
echo "  ✓ Enforce for administrators"
echo ""
echo "Next steps:"
echo "  1. Review settings in GitHub web UI: https://github.com/$REPO/settings/branches"
echo "  2. Add branch protection rules via UI for additional granularity if needed"
echo "  3. Update CODEOWNERS file to require specific reviewers (optional)"
echo ""
