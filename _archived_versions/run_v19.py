#!/usr/bin/env python3
"""
V19.0 Main Runner
==================
Run the complete V19.0 Reversal-Ensemble System.

Usage:
    python run_v19.py           # Run all phases
    python run_v19.py phase1    # Run only reversal
    python run_v19.py phase2    # Run only mean reversion
    python run_v19.py phase3    # Run only ensemble
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V19_Runner')


def main():
    logger.info("=" * 70)
    logger.info("🚀 V19.0 REVERSAL-ENSEMBLE SYSTEM")
    logger.info("=" * 70)
    
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    results_dir = Path('results/v19')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if phase in ['all', 'phase1']:
        logger.info("\n" + "=" * 70)
        logger.info("📊 PHASE 1: PURE REVERSAL STRATEGY")
        logger.info("=" * 70)
        from v19_reversal_strategy import run_reversal_backtest
        p1_result = run_reversal_backtest()
        
        if phase == 'phase1':
            return
    
    if phase in ['all', 'phase2']:
        logger.info("\n" + "=" * 70)
        logger.info("📊 PHASE 2: MEAN REVERSION STRATEGY")
        logger.info("=" * 70)
        from v19_mean_reversion import run_meanrev_backtest
        p2_result = run_meanrev_backtest()
        
        if phase == 'phase2':
            return
    
    if phase in ['all', 'phase3']:
        logger.info("\n" + "=" * 70)
        logger.info("📊 PHASE 3: ENSEMBLE SYSTEM")
        logger.info("=" * 70)
        from v19_ensemble import run_ensemble_backtest
        ensemble_result = run_ensemble_backtest()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📋 V19.0 FINAL SUMMARY")
    logger.info("=" * 70)
    
    import json
    
    try:
        with open(results_dir / 'v19_reversal_results.json') as f:
            p1 = json.load(f)
        with open(results_dir / 'v19_meanrev_results.json') as f:
            p2 = json.load(f)
        with open(results_dir / 'v19_ensemble_results.json') as f:
            p3 = json.load(f)
        
        logger.info("\n📈 Performance Comparison:")
        logger.info(f"{'Strategy':<20} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10} {'WinRate':>10}")
        logger.info("-" * 60)
        logger.info(f"{'Reversal':<20} {p1['cagr']:>10.1%} {p1['sharpe']:>10.2f} {p1['max_drawdown']:>10.1%} {p1['win_rate']:>10.1%}")
        logger.info(f"{'Mean Reversion':<20} {p2['cagr']:>10.1%} {p2['sharpe']:>10.2f} {p2['max_drawdown']:>10.1%} {p2['win_rate']:>10.1%}")
        logger.info(f"{'Ensemble':<20} {p3['cagr']:>10.1%} {p3['sharpe']:>10.2f} {p3['max_drawdown']:>10.1%} {p3['win_rate']:>10.1%}")
        
        logger.info("\n🎯 Target Achievement:")
        all_pass = all(p3['targets_met'].values())
        for target, met in p3['targets_met'].items():
            status = "✅ PASS" if met else "❌ FAIL"
            logger.info(f"   {target}: {status}")
        
        if all_pass:
            logger.info("\n" + "=" * 70)
            logger.info("🎉 ALL TARGETS MET! SYSTEM READY FOR PAPER TRADING")
            logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Could not load results: {e}")
    
    logger.info(f"\n📄 Reports saved to {results_dir}/")


if __name__ == "__main__":
    main()
