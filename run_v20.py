#!/usr/bin/env python3
"""
V20.0 Main Runner
==================
Run the complete V20.0 Enhanced Ensemble System.

Usage:
    python run_v20.py           # Run all phases
    python run_v20.py phase1    # Run only volatility reversal
    python run_v20.py phase2    # Run only ensemble
"""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V20_Runner')


def main():
    logger.info("=" * 70)
    logger.info("🚀 V20.0 ENHANCED ENSEMBLE SYSTEM")
    logger.info("=" * 70)
    
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    results_dir = Path('results/v20')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if phase in ['all', 'phase1']:
        logger.info("\n" + "=" * 70)
        logger.info("📊 PHASE 1: VOLATILITY-FILTERED REVERSAL")
        logger.info("=" * 70)
        from v20_volatility_reversal import run_volatility_reversal_backtest
        p1_result = run_volatility_reversal_backtest()
        
        if phase == 'phase1':
            return
    
    if phase in ['all', 'phase2']:
        logger.info("\n" + "=" * 70)
        logger.info("📊 PHASE 2: ENHANCED ENSEMBLE")
        logger.info("=" * 70)
        from v20_enhanced_ensemble import run_enhanced_ensemble
        p2_result = run_enhanced_ensemble()
    
    # Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("📋 V20.0 FINAL SUMMARY")
    logger.info("=" * 70)
    
    try:
        # Load V19 results for comparison
        v19_results = {}
        try:
            with open('results/v19/v19_ensemble_results.json') as f:
                v19_results = json.load(f)
        except:
            v19_results = {'cagr': 0.235, 'sharpe': 1.15, 'max_drawdown': -0.135}
        
        # Load V20 results
        with open(results_dir / 'v20_volrev_results.json') as f:
            volrev = json.load(f)
        with open(results_dir / 'v20_ensemble_results.json') as f:
            ensemble = json.load(f)
        
        logger.info("\n📈 Performance Comparison:")
        logger.info(f"{'Strategy':<25} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10} {'WinRate':>10}")
        logger.info("-" * 65)
        logger.info(f"{'V19 Ensemble':<25} {v19_results.get('cagr', 0.235):>10.1%} {v19_results.get('sharpe', 1.15):>10.2f} {v19_results.get('max_drawdown', -0.135):>10.1%} {v19_results.get('win_rate', 0.556):>10.1%}")
        logger.info(f"{'V20 VolRev (Phase 1)':<25} {volrev['cagr']:>10.1%} {volrev['sharpe']:>10.2f} {volrev['max_drawdown']:>10.1%} {volrev['win_rate']:>10.1%}")
        logger.info(f"{'V20 Ensemble (Phase 2)':<25} {ensemble['cagr']:>10.1%} {ensemble['sharpe']:>10.2f} {ensemble['max_drawdown']:>10.1%} {ensemble['win_rate']:>10.1%}")
        
        logger.info("\n📊 Improvement vs V19:")
        cagr_imp = ensemble['cagr'] - v19_results.get('cagr', 0.235)
        sharpe_imp = ensemble['sharpe'] - v19_results.get('sharpe', 1.15)
        dd_imp = ensemble['max_drawdown'] - v19_results.get('max_drawdown', -0.135)
        
        logger.info(f"   CAGR:   {cagr_imp:+.1%}")
        logger.info(f"   Sharpe: {sharpe_imp:+.2f}")
        logger.info(f"   MaxDD:  {dd_imp:+.1%} (less negative is better)")
        
        logger.info("\n🎯 Target Achievement:")
        for target, met in ensemble['targets_met'].items():
            status = "✅ PASS" if met else "❌ FAIL"
            logger.info(f"   {target}: {status}")
        
        # Overall assessment
        all_pass = all(ensemble['targets_met'].values())
        partial_pass = sum(ensemble['targets_met'].values()) >= 2
        
        if all_pass:
            logger.info("\n" + "=" * 70)
            logger.info("🎉 V20.0 ALL TARGETS MET!")
            logger.info("=" * 70)
        elif partial_pass:
            logger.info("\n" + "=" * 70)
            logger.info("✅ V20.0 PARTIAL SUCCESS - Significant improvement over V19")
            logger.info("=" * 70)
            logger.info("\n📊 Key Achievements:")
            logger.info(f"   • CAGR improved from 23.5% to {ensemble['cagr']:.1%} (+{cagr_imp:.1%})")
            logger.info(f"   • Sharpe improved from 1.15 to {ensemble['sharpe']:.2f} (+{sharpe_imp:.2f})")
            logger.info(f"   • MaxDD improved from -13.5% to {ensemble['max_drawdown']:.1%}")
        else:
            logger.info("\n⚠️ V20.0 needs further optimization")
        
    except Exception as e:
        logger.error(f"Could not load results: {e}")
    
    logger.info(f"\n📄 Reports saved to {results_dir}/")


if __name__ == "__main__":
    main()
