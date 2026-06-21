#!/bin/bash
# Pre-commit hook: validate promotion gates before local commits
# 
# Installation:
#   cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# This hook:
#   - Detects staged promotion gate evidence files
#   - Validates them against gate thresholds
#   - Blocks commit if validation fails
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}[pre-commit] Checking for promotion gate evidence...${NC}"

# Get staged files
STAGED=$(git diff --cached --name-only)

# Check for promotion evidence files
RESEARCH_PAPER=$(echo "$STAGED" | grep -E 'promotion_gate_evidence.*research_to_paper' || true)
PAPER_LIVE=$(echo "$STAGED" | grep -E 'promotion_gate_evidence.*paper_to_live' || true)
LIVE_SCALEUP=$(echo "$STAGED" | grep -E 'promotion_gate_evidence.*live_scale_up' || true)

if [ -z "$RESEARCH_PAPER" ] && [ -z "$PAPER_LIVE" ] && [ -z "$LIVE_SCALEUP" ]; then
    echo -e "${GREEN}[pre-commit] No promotion evidence files detected. Proceeding.${NC}"
    exit 0
fi

echo -e "${YELLOW}[pre-commit] Promotion evidence detected. Validating...${NC}"

# Validate each detected gate
if [ -n "$RESEARCH_PAPER" ]; then
    echo "Validating research->paper gate: $RESEARCH_PAPER"
    if ! python scripts/check_promotion_gates.py --gate research_to_paper --input "$RESEARCH_PAPER" > /tmp/rp_gate.log 2>&1; then
        echo -e "${RED}[pre-commit] research->paper gate FAILED${NC}"
        cat /tmp/rp_gate.log
        exit 1
    fi
    echo -e "${GREEN}[pre-commit] research->paper gate PASSED${NC}"
fi

if [ -n "$PAPER_LIVE" ]; then
    echo "Validating paper->live gate: $PAPER_LIVE"
    if ! python scripts/check_promotion_gates.py --gate paper_to_live --input "$PAPER_LIVE" > /tmp/pl_gate.log 2>&1; then
        echo -e "${RED}[pre-commit] paper->live gate FAILED${NC}"
        cat /tmp/pl_gate.log
        exit 1
    fi
    echo -e "${GREEN}[pre-commit] paper->live gate PASSED${NC}"
fi

if [ -n "$LIVE_SCALEUP" ]; then
    echo "Validating live scale-up gate: $LIVE_SCALEUP"
    if ! python scripts/check_promotion_gates.py --gate live_scale_up --input "$LIVE_SCALEUP" > /tmp/ls_gate.log 2>&1; then
        echo -e "${RED}[pre-commit] live scale-up gate FAILED${NC}"
        cat /tmp/ls_gate.log
        exit 1
    fi
    echo -e "${GREEN}[pre-commit] live scale-up gate PASSED${NC}"
fi

echo -e "${GREEN}[pre-commit] All promotion gates passed. Proceeding with commit.${NC}"
exit 0
