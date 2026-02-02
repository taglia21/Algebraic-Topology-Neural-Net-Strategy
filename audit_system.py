#!/usr/bin/env python3
"""Comprehensive System Audit - Identifies all points of failure."""

import os
import sys
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

class SystemAuditor:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.critical = []
        
    def audit_file(self, filepath: str) -> Dict:
        """Audit a single Python file."""
        issues = []
        
        # Check if file exists
        if not os.path.exists(filepath):
            return {'file': filepath, 'status': 'MISSING', 'issues': ['File does not exist']}
        
        # Check syntax
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"SYNTAX ERROR: {e}")
            return {'file': filepath, 'status': 'SYNTAX_ERROR', 'issues': issues}
        
        # Check for common issues
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Check for TODOs/FIXMEs
        for i, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line:
                issues.append(f"Line {i}: {line.strip()}")
        
        return {'file': filepath, 'status': 'OK', 'issues': issues}
    
    def check_imports(self, filepath: str) -> List[str]:
        """Check if all imports are available."""
        missing = []
        try:
            result = subprocess.run(
                ['python', '-m', 'py_compile', filepath],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                missing.append(result.stderr)
        except Exception as e:
            missing.append(str(e))
        return missing
    
    def audit_paper_trading_bot(self):
        """Audit the main paper trading bot."""
        print("\n=== AUDITING: paper_trading_bot.py ===")
        
        if not os.path.exists('paper_trading_bot.py'):
            self.critical.append("paper_trading_bot.py: FILE MISSING")
            return
        
        result = self.audit_file('paper_trading_bot.py')
        print(f"Status: {result['status']}")
        
        # Check for critical imports
        with open('paper_trading_bot.py') as f:
            content = f.read()
            
        issues_found = []
        
        # Check if TDAStrategy is used but might not have generate_signals()
        if 'TDAStrategy' in content and 'generate_signals' in content:
            issues_found.append("WARNING: TDAStrategy.generate_signals() may not be implemented")
        
        # Check if signal execution logic exists
        if 'execute_signals' in content:
            if 'assumed_price = 100.0' in content:
                issues_found.append("CRITICAL: Using hardcoded price (100.0) instead of real market prices")
        
        for issue in issues_found:
            print(f"  - {issue}")
            self.issues.append(f"paper_trading_bot.py: {issue}")
    
    def audit_live_executor(self):
        """Audit the live executor."""
        print("\n=== AUDITING: src/execution/live_executor.py ===")
        
        filepath = 'src/execution/live_executor.py'
        if not os.path.exists(filepath):
            self.critical.append(f"{filepath}: FILE MISSING")
            return
        
        with open(filepath) as f:
            content = f.read()
        
        issues_found = []
        
        # Check for hardcoded prices
        if 'fill_price = order.limit_price if order.limit_price else 100.0' in content:
            issues_found.append("CRITICAL: Paper trading uses placeholder price (100.0) - no real market data integration")
        
        # Check if real TDA integration exists
        if 'TODO: Implement real TDA API execution' in content:
            issues_found.append("CRITICAL: Live TDA execution not implemented")
        
        # Check position valuation
        if 'def get_account_value' in content:
            if 'TODO' in content:
                issues_found.append("WARNING: Account value calculation incomplete")
        
        for issue in issues_found:
            print(f"  - {issue}")
            self.issues.append(f"live_executor.py: {issue}")
    
    def audit_tda_strategy(self):
        """Audit TDA strategy."""
        print("\n=== AUDITING: src/tda_strategy.py ===")
        
        filepath = 'src/tda_strategy.py'
        if not os.path.exists(filepath):
            self.critical.append(f"{filepath}: FILE MISSING")
            return
        
        with open(filepath) as f:
            content = f.read()
        
        issues_found = []
        
        # Check if generate_signals exists
        if 'def generate_signals' not in content:
            issues_found.append("CRITICAL: generate_signals() method not found - paper trading bot will fail")
        
        # Check for real strategy logic
        if 'yfinance' in content:
            issues_found.append("WARNING: Using yfinance (15-min delayed data)")
        
        for issue in issues_found:
            print(f"  - {issue}")
            self.issues.append(f"tda_strategy.py: {issue}")
    
    def run_full_audit(self):
        """Run complete system audit."""
        print("=" * 70)
        print("COMPREHENSIVE SYSTEM AUDIT")
        print("=" * 70)
        
        self.audit_paper_trading_bot()
        self.audit_live_executor()
        self.audit_tda_strategy()
        
        print("\n" + "=" * 70)
        print("AUDIT SUMMARY")
        print("=" * 70)
        
        print(f"\nCRITICAL ISSUES: {len(self.critical)}")
        for issue in self.critical:
            print(f"  ❌ {issue}")
        
        print(f"\nWARNINGS & ISSUES: {len(self.issues)}")
        for issue in self.issues:
            print(f"  ⚠️  {issue}")
        
        print("\n" + "=" * 70)
        return len(self.critical) + len(self.issues)

if __name__ == '__main__':
    auditor = SystemAuditor()
    total_issues = auditor.run_full_audit()
    sys.exit(0 if total_issues == 0 else 1)
