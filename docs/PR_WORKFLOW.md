# PR-First Workflow and Branch Protection

This repository previously absorbed production changes by direct pushes to `main`.
Use this workflow to preserve code review, auditability, and rollback safety.

## Required Branch Workflow

1. Sync local main:

```bash
git checkout main
git pull origin main
```

2. Create a feature branch from `main`:

```bash
git checkout -b feat/<short-scope-name>
```

3. Make changes, run validation, and commit in logical chunks:

```bash
python -m pytest tests/ -v
git add -A
git commit -m "<scope>: <summary>"
```

4. Push branch and create PR:

```bash
git push -u origin feat/<short-scope-name>
gh pr create --base main --head feat/<short-scope-name> --title "<title>" --body-file <pr_body.md>
```

5. Merge only after review and CI pass.

## GitHub Branch Protection Settings (`main`)

Enable these controls in repository settings:

1. Require a pull request before merging.
2. Require at least 1 approving review.
3. Dismiss stale approvals when new commits are pushed.
4. Require status checks to pass before merging.
5. Require conversation resolution before merging.
6. Restrict who can push to matching branches.
7. Include administrators in protection.
8. Disable force pushes and branch deletions.

## Optional Local Guardrail (pre-push)

Add a local Git hook to prevent accidental pushes to `main`:

```bash
cat > .git/hooks/pre-push << 'EOF'
#!/usr/bin/env bash
branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "main" ]]; then
  echo "Push blocked: direct pushes to main are disabled. Use a feature branch + PR."
  exit 1
fi
EOF
chmod +x .git/hooks/pre-push
```

Note: Local hooks are not shared automatically. Branch protection is the source of truth.
