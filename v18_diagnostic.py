#!/usr/bin/env python3
"""
V18.0 Diagnostic Analysis
==========================
Root cause analysis for V17.0's -12.8% CAGR.

Answers:
1. What % of loss comes from each regime?
2. What % of loss comes from transaction costs vs bad signals?
3. Which factors have NEGATIVE IC? (hurting performance)
4. Is the HMM detecting regimes correctly?

Output: results/v18/DIAGNOSTIC_REPORT.md
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V18_Diagnostic')


class V18Diagnostic:
    """Comprehensive diagnostic for V17.0 performance issues"""
    
    def __init__(self):
        self.prices = None
        self.factors = None
        self.regime_history = None
        self.equity_curve = None
        self.findings = {}
        
    def load_data(self):
        """Load all V17 data"""
        logger.info("📂 Loading V17 data...")
        
        # Load prices
        price_file = 'cache/v17_prices/v17_prices_latest.parquet'
        if os.path.exists(price_file):
            self.prices = pd.read_parquet(price_file)
            self.prices['date'] = pd.to_datetime(self.prices['date'])
            logger.info(f"   Prices: {len(self.prices)} rows, {self.prices['symbol'].nunique()} symbols")
        
        # Load equity curve
        equity_file = 'results/v17/v17_equity_curve.parquet'
        if os.path.exists(equity_file):
            self.equity_curve = pd.read_parquet(equity_file)
            self.equity_curve['date'] = pd.to_datetime(self.equity_curve['date'])
            logger.info(f"   Equity curve: {len(self.equity_curve)} days")
        
        # Load regime history
        regime_file = 'cache/v17_regime_history.parquet'
        if os.path.exists(regime_file):
            self.regime_history = pd.read_parquet(regime_file)
            if 'date' in self.regime_history.columns:
                self.regime_history['date'] = pd.to_datetime(self.regime_history['date'])
            logger.info(f"   Regime history: {len(self.regime_history)} days")
        
        # Load V17 backtest results
        results_file = 'results/v17/v17_full_results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.v17_results = json.load(f)
            logger.info(f"   V17 results loaded")
    
    def analyze_regime_attribution(self) -> Dict:
        """
        Question 1: What % of loss comes from each regime?
        """
        logger.info("\n📊 Analyzing regime attribution...")
        
        if self.equity_curve is None or self.regime_history is None:
            logger.warning("   Missing equity curve or regime history")
            return {}
        
        # Merge equity with regime
        eq = self.equity_curve.copy()
        eq['return'] = eq['portfolio_value'].pct_change()
        
        # Get SPY returns for regime assignment
        spy_data = self.prices[self.prices['symbol'] == 'SPY'].copy()
        if spy_data.empty:
            # Fetch SPY
            spy = yf.download('SPY', start='2024-01-01', progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [c.lower() for c in spy.columns]
            spy = spy.reset_index()
            spy.columns = [c.lower() if c != 'Date' else 'date' for c in spy.columns]
            spy_data = spy
        
        spy_data['date'] = pd.to_datetime(spy_data['date'])
        
        # Merge regime with equity curve
        regime_daily = self.regime_history[['date', 'regime', 'regime_name']].drop_duplicates()
        eq = eq.merge(regime_daily, on='date', how='left')
        eq['regime'] = eq['regime'].fillna(2)  # Default to MeanRevert
        eq['regime_name'] = eq['regime_name'].fillna('LowVolMeanRevert')
        
        # Calculate P&L by regime
        regime_pnl = {}
        total_pnl = eq['return'].sum()
        
        for regime in [0, 1, 2, 3]:
            regime_mask = eq['regime'] == regime
            regime_returns = eq.loc[regime_mask, 'return']
            regime_pnl[regime] = {
                'name': eq.loc[regime_mask, 'regime_name'].iloc[0] if regime_mask.any() else f'Regime_{regime}',
                'days': regime_mask.sum(),
                'total_return': regime_returns.sum(),
                'avg_daily_return': regime_returns.mean() * 100 if len(regime_returns) > 0 else 0,
                'pct_of_total_loss': (regime_returns.sum() / total_pnl * 100) if total_pnl != 0 else 0
            }
        
        self.findings['regime_attribution'] = regime_pnl
        
        logger.info(f"   Total Return: {total_pnl:.4f}")
        for regime, data in regime_pnl.items():
            logger.info(f"   Regime {regime} ({data['name']}): {data['total_return']:.4f} ({data['pct_of_total_loss']:.1f}%)")
        
        return regime_pnl
    
    def analyze_cost_vs_signal(self) -> Dict:
        """
        Question 2: What % of loss comes from transaction costs vs bad signals?
        """
        logger.info("\n💰 Analyzing cost vs signal attribution...")
        
        if not hasattr(self, 'v17_results'):
            logger.warning("   Missing V17 results")
            return {}
        
        results = self.v17_results
        
        # Get key metrics
        total_return = results.get('total_return', 0)
        total_costs = results.get('total_costs', 0)
        initial_capital = 1_000_000  # From config
        cost_drag = results.get('cost_drag', 0)
        
        # If we had no costs, what would return be?
        gross_return = total_return + cost_drag
        
        # Attribution
        cost_attribution = {
            'total_return': total_return,
            'gross_return_before_costs': gross_return,
            'total_transaction_costs': total_costs,
            'cost_as_pct_of_capital': cost_drag * 100,
            'signal_pnl': gross_return * initial_capital,
            'signal_is_profitable': gross_return > 0,
            'cost_impact_pct': (cost_drag / abs(total_return) * 100) if total_return != 0 else 0
        }
        
        # How much of the loss is from costs vs signals?
        if total_return < 0:
            loss_from_costs = min(cost_drag, abs(total_return))
            loss_from_signals = abs(total_return) - loss_from_costs
            cost_attribution['pct_loss_from_costs'] = (loss_from_costs / abs(total_return)) * 100
            cost_attribution['pct_loss_from_signals'] = (loss_from_signals / abs(total_return)) * 100
        else:
            cost_attribution['pct_loss_from_costs'] = 0
            cost_attribution['pct_loss_from_signals'] = 0
        
        self.findings['cost_attribution'] = cost_attribution
        
        logger.info(f"   Total Return: {total_return:.2%}")
        logger.info(f"   Gross Return (before costs): {gross_return:.2%}")
        logger.info(f"   Transaction Costs: ${total_costs:,.0f} ({cost_drag:.2%})")
        if total_return < 0:
            logger.info(f"   Loss from costs: {cost_attribution['pct_loss_from_costs']:.1f}%")
            logger.info(f"   Loss from signals: {cost_attribution['pct_loss_from_signals']:.1f}%")
        
        return cost_attribution
    
    def calculate_factor_ic(self) -> pd.DataFrame:
        """
        Question 3: Which factors have NEGATIVE IC?
        
        Information Coefficient = Correlation between factor value and forward returns
        """
        logger.info("\n📈 Calculating Factor Information Coefficients...")
        
        if self.prices is None:
            logger.warning("   Missing price data")
            return pd.DataFrame()
        
        # Import factor zoo
        from v17_factor_zoo import FactorZoo
        zoo = FactorZoo()
        
        # Get list of all factors
        all_factors = zoo.get_all_factors()
        
        # Calculate factors and forward returns for a sample of symbols
        sample_symbols = self.prices['symbol'].unique()[:100]  # Sample 100 symbols
        
        ic_results = []
        
        logger.info(f"   Calculating IC for {len(all_factors)} factors across {len(sample_symbols)} symbols...")
        
        for symbol in sample_symbols:
            sym_data = self.prices[self.prices['symbol'] == symbol].copy()
            
            if len(sym_data) < 100:
                continue
            
            sym_data = sym_data.sort_values('date')
            
            # Calculate forward returns (1-day, 5-day, 20-day)
            sym_data['fwd_ret_1d'] = sym_data['close'].pct_change(1).shift(-1)
            sym_data['fwd_ret_5d'] = sym_data['close'].pct_change(5).shift(-5)
            sym_data['fwd_ret_20d'] = sym_data['close'].pct_change(20).shift(-20)
            
            # Calculate all factors
            try:
                factor_df = zoo.compute_all_factors(sym_data)
            except Exception as e:
                continue
            
            # Merge
            sym_data = sym_data.merge(
                factor_df.drop(columns=['close'], errors='ignore'),
                left_index=True, right_index=True, how='left'
            )
            
            # Calculate IC for each factor
            for factor in all_factors:
                if factor not in sym_data.columns:
                    continue
                
                # 5-day forward return IC (most relevant)
                valid = sym_data[[factor, 'fwd_ret_5d']].dropna()
                if len(valid) < 50:
                    continue
                
                try:
                    ic, p_val = stats.spearmanr(valid[factor], valid['fwd_ret_5d'])
                    
                    ic_results.append({
                        'symbol': symbol,
                        'factor': factor,
                        'ic_5d': ic,
                        'p_value': p_val,
                        'n_obs': len(valid)
                    })
                except:
                    continue
        
        if not ic_results:
            logger.warning("   No IC results calculated")
            return pd.DataFrame()
        
        ic_df = pd.DataFrame(ic_results)
        
        # Aggregate IC by factor (mean across symbols)
        factor_ic = ic_df.groupby('factor').agg({
            'ic_5d': ['mean', 'std', 'count'],
            'p_value': 'mean'
        }).reset_index()
        
        factor_ic.columns = ['factor', 'mean_ic', 'std_ic', 'n_symbols', 'avg_pval']
        factor_ic['t_stat'] = factor_ic['mean_ic'] / (factor_ic['std_ic'] / np.sqrt(factor_ic['n_symbols']))
        factor_ic['significant'] = factor_ic['avg_pval'] < 0.05
        factor_ic = factor_ic.sort_values('mean_ic', ascending=True)
        
        self.findings['factor_ic'] = factor_ic
        
        # Count negative IC factors
        negative_ic = factor_ic[factor_ic['mean_ic'] < 0]
        positive_ic = factor_ic[factor_ic['mean_ic'] > 0]
        
        logger.info(f"   Factors with NEGATIVE IC: {len(negative_ic)}")
        logger.info(f"   Factors with POSITIVE IC: {len(positive_ic)}")
        
        # Top 5 worst
        logger.info("   Top 5 WORST factors (most negative IC):")
        for _, row in negative_ic.head(5).iterrows():
            logger.info(f"      {row['factor']}: IC={row['mean_ic']:.4f}")
        
        # Top 5 best
        logger.info("   Top 5 BEST factors (most positive IC):")
        for _, row in positive_ic.tail(5).iterrows():
            logger.info(f"      {row['factor']}: IC={row['mean_ic']:.4f}")
        
        return factor_ic
    
    def validate_hmm_regime(self) -> Dict:
        """
        Question 4: Is the HMM detecting regimes correctly?
        
        Compare HMM regime to VIX levels and SPY drawdowns
        """
        logger.info("\n🧠 Validating HMM Regime Detection...")
        
        # Fetch VIX
        logger.info("   Fetching VIX data...")
        vix = yf.download('^VIX', start='2024-01-01', progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix.columns = [c.lower() for c in vix.columns]
        vix = vix.reset_index()
        vix.columns = [c.lower() if c != 'Date' else 'date' for c in vix.columns]
        vix['date'] = pd.to_datetime(vix['date'])
        
        if self.regime_history is None:
            logger.warning("   Missing regime history")
            return {}
        
        # Merge regime with VIX
        regime_daily = self.regime_history[['date', 'regime', 'regime_name']].drop_duplicates()
        merged = regime_daily.merge(vix[['date', 'close']], on='date', how='inner')
        merged = merged.rename(columns={'close': 'vix'})
        
        # Expected behavior:
        # - Crisis (regime 3) should have HIGH VIX (>25)
        # - LowVolTrend (regime 0) should have LOW VIX (<15)
        
        validation = {}
        
        for regime in [0, 1, 2, 3]:
            regime_mask = merged['regime'] == regime
            if not regime_mask.any():
                continue
            
            regime_vix = merged.loc[regime_mask, 'vix']
            regime_name = merged.loc[regime_mask, 'regime_name'].iloc[0]
            
            validation[regime] = {
                'name': regime_name,
                'n_days': regime_mask.sum(),
                'avg_vix': regime_vix.mean(),
                'median_vix': regime_vix.median(),
                'max_vix': regime_vix.max(),
                'min_vix': regime_vix.min()
            }
        
        # Check if regime ordering makes sense
        # Crisis should have highest VIX, LowVolTrend lowest
        issues = []
        
        if 0 in validation and 3 in validation:
            if validation[0]['avg_vix'] > validation[3]['avg_vix']:
                issues.append("ISSUE: LowVolTrend has HIGHER VIX than Crisis!")
        
        if 3 in validation and validation[3]['avg_vix'] < 20:
            issues.append(f"ISSUE: Crisis regime has low VIX ({validation[3]['avg_vix']:.1f})")
        
        if 0 in validation and validation[0]['avg_vix'] > 20:
            issues.append(f"ISSUE: LowVolTrend has high VIX ({validation[0]['avg_vix']:.1f})")
        
        validation['issues'] = issues
        validation['hmm_is_valid'] = len(issues) == 0
        
        self.findings['hmm_validation'] = validation
        
        logger.info(f"   VIX by Regime:")
        for regime in [0, 1, 2, 3]:
            if regime in validation:
                v = validation[regime]
                logger.info(f"      Regime {regime} ({v['name']}): VIX avg={v['avg_vix']:.1f}, days={v['n_days']}")
        
        if issues:
            logger.warning("   ⚠️ HMM ISSUES DETECTED:")
            for issue in issues:
                logger.warning(f"      {issue}")
        else:
            logger.info("   ✅ HMM regime detection looks valid")
        
        return validation
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        logger.info("\n📝 Generating Diagnostic Report...")
        
        # Create output directory
        output_dir = Path('results/v18')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build report
        report = self._build_markdown_report()
        
        # Save report
        report_path = output_dir / 'DIAGNOSTIC_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save raw findings as JSON
        findings_path = output_dir / 'diagnostic_findings.json'
        
        # Convert DataFrames to dicts for JSON
        json_findings = {}
        for key, value in self.findings.items():
            if isinstance(value, pd.DataFrame):
                json_findings[key] = value.to_dict(orient='records')
            else:
                json_findings[key] = value
        
        with open(findings_path, 'w') as f:
            json.dump(json_findings, f, indent=2, default=str)
        
        logger.info(f"   ✅ Report saved to {report_path}")
        
        return report
    
    def _build_markdown_report(self) -> str:
        """Build the markdown diagnostic report"""
        
        report = f"""# V18.0 Diagnostic Report

**Generated:** {datetime.now():%Y-%m-%d %H:%M:%S}
**Analyzing:** V17.0 Performance (-12.8% CAGR)

---

## Executive Summary

### Top 3 Issues Identified

"""
        # Determine top 3 issues
        issues = []
        
        # Issue 1: Check cost vs signal
        if 'cost_attribution' in self.findings:
            ca = self.findings['cost_attribution']
            if ca.get('pct_loss_from_signals', 0) > 50:
                issues.append(f"1. **BAD SIGNALS**: {ca['pct_loss_from_signals']:.1f}% of loss from signal quality (not costs)")
            else:
                issues.append(f"1. **COST DRAG**: Transaction costs account for {ca['pct_loss_from_costs']:.1f}% of loss")
        
        # Issue 2: Check factor IC
        if 'factor_ic' in self.findings:
            ic_df = self.findings['factor_ic']
            if isinstance(ic_df, pd.DataFrame):
                neg_ic_count = (ic_df['mean_ic'] < 0).sum()
                if neg_ic_count > 25:  # More than half
                    issues.append(f"2. **NEGATIVE IC FACTORS**: {neg_ic_count}/{len(ic_df)} factors have NEGATIVE IC (hurting performance)")
                else:
                    issues.append(f"2. **FACTOR QUALITY**: {neg_ic_count} factors with negative IC need removal/flipping")
        
        # Issue 3: Check HMM
        if 'hmm_validation' in self.findings:
            hv = self.findings['hmm_validation']
            if not hv.get('hmm_is_valid', True):
                issues.append(f"3. **HMM MISALIGNMENT**: {len(hv.get('issues', []))} regime detection issues found")
            else:
                issues.append("3. **HMM OK**: Regime detection validated against VIX")
        
        # Issue 4: Regime attribution
        if 'regime_attribution' in self.findings:
            ra = self.findings['regime_attribution']
            worst_regime = max(ra.items(), key=lambda x: abs(x[1].get('total_return', 0)) if x[1].get('total_return', 0) < 0 else 0)
            if worst_regime[1].get('total_return', 0) < 0:
                issues.append(f"4. **REGIME CONCENTRATION**: Regime {worst_regime[0]} ({worst_regime[1]['name']}) caused {worst_regime[1]['pct_of_total_loss']:.1f}% of losses")
        
        for issue in issues[:3]:
            report += f"{issue}\n\n"
        
        report += """
---

## 1. Regime Attribution Analysis

**Question:** What % of loss comes from each regime?

"""
        if 'regime_attribution' in self.findings:
            ra = self.findings['regime_attribution']
            report += "| Regime | Name | Days | Total Return | % of Total P&L |\n"
            report += "|--------|------|------|--------------|----------------|\n"
            
            for regime in [0, 1, 2, 3]:
                if regime in ra:
                    r = ra[regime]
                    report += f"| {regime} | {r['name']} | {r['days']} | {r['total_return']:.4f} | {r['pct_of_total_loss']:.1f}% |\n"
        
        report += """

---

## 2. Transaction Cost vs Signal Quality

**Question:** What % of loss comes from costs vs bad signals?

"""
        if 'cost_attribution' in self.findings:
            ca = self.findings['cost_attribution']
            report += f"""
| Metric | Value |
|--------|-------|
| Total Return | {ca['total_return']:.2%} |
| Gross Return (before costs) | {ca['gross_return_before_costs']:.2%} |
| Transaction Costs | ${ca['total_transaction_costs']:,.0f} |
| Cost Drag | {ca['cost_as_pct_of_capital']:.2f}% of capital |
| **% Loss from Costs** | {ca.get('pct_loss_from_costs', 0):.1f}% |
| **% Loss from Signals** | {ca.get('pct_loss_from_signals', 0):.1f}% |

**Conclusion:** {"Signals are the main problem" if ca.get('pct_loss_from_signals', 0) > 50 else "Costs are a significant drag"}
"""
        
        report += """

---

## 3. Factor Information Coefficient (IC) Analysis

**Question:** Which factors have NEGATIVE IC?

A negative IC means the factor is **inversely** correlated with forward returns - 
we're using it backwards or it's spurious.

"""
        if 'factor_ic' in self.findings:
            ic_df = self.findings['factor_ic']
            if isinstance(ic_df, pd.DataFrame) and not ic_df.empty:
                neg_ic = ic_df[ic_df['mean_ic'] < 0].sort_values('mean_ic')
                pos_ic = ic_df[ic_df['mean_ic'] > 0].sort_values('mean_ic', ascending=False)
                
                report += f"**Summary:** {len(neg_ic)} negative IC factors, {len(pos_ic)} positive IC factors\n\n"
                
                report += "### Factors with NEGATIVE IC (REMOVE OR FLIP)\n\n"
                report += "| Factor | Mean IC | Std IC | T-Stat | Action |\n"
                report += "|--------|---------|--------|--------|--------|\n"
                
                for _, row in neg_ic.iterrows():
                    action = "FLIP" if abs(row['mean_ic']) > 0.02 else "REMOVE"
                    report += f"| {row['factor']} | {row['mean_ic']:.4f} | {row['std_ic']:.4f} | {row['t_stat']:.2f} | {action} |\n"
                
                report += "\n### Factors with POSITIVE IC (KEEP)\n\n"
                report += "| Factor | Mean IC | Std IC | T-Stat | Significant |\n"
                report += "|--------|---------|--------|--------|-------------|\n"
                
                for _, row in pos_ic.head(20).iterrows():
                    sig = "✓" if row['significant'] else ""
                    report += f"| {row['factor']} | {row['mean_ic']:.4f} | {row['std_ic']:.4f} | {row['t_stat']:.2f} | {sig} |\n"
                
                if len(pos_ic) > 20:
                    report += f"\n*...and {len(pos_ic) - 20} more positive IC factors*\n"
        
        report += """

---

## 4. HMM Regime Validation

**Question:** Is the HMM detecting regimes correctly?

We compare HMM regime assignments to VIX levels:
- **Crisis** should have HIGH VIX (>25)
- **LowVolTrend** should have LOW VIX (<15)

"""
        if 'hmm_validation' in self.findings:
            hv = self.findings['hmm_validation']
            
            report += "| Regime | Name | Days | Avg VIX | Median VIX | Max VIX |\n"
            report += "|--------|------|------|---------|------------|--------|\n"
            
            for regime in [0, 1, 2, 3]:
                if regime in hv and isinstance(hv[regime], dict):
                    v = hv[regime]
                    report += f"| {regime} | {v['name']} | {v['n_days']} | {v['avg_vix']:.1f} | {v['median_vix']:.1f} | {v['max_vix']:.1f} |\n"
            
            if hv.get('issues'):
                report += "\n### ⚠️ Issues Detected\n\n"
                for issue in hv['issues']:
                    report += f"- {issue}\n"
            else:
                report += "\n### ✅ Validation Passed\n\nHMM regime detection aligns with VIX behavior.\n"
        
        report += """

---

## Recommendations

Based on the diagnostic findings:

"""
        # Generate recommendations based on findings
        recommendations = []
        
        if 'factor_ic' in self.findings:
            ic_df = self.findings['factor_ic']
            if isinstance(ic_df, pd.DataFrame):
                neg_count = (ic_df['mean_ic'] < 0).sum()
                if neg_count > 10:
                    recommendations.append(f"1. **Remove/Flip {neg_count} negative IC factors** - These are hurting performance")
        
        if 'cost_attribution' in self.findings:
            ca = self.findings['cost_attribution']
            if ca.get('pct_loss_from_signals', 0) > 80:
                recommendations.append("2. **Focus on signal quality** - Costs are not the main issue")
            else:
                recommendations.append("2. **Reduce turnover** - High transaction costs are dragging performance")
        
        if 'regime_attribution' in self.findings:
            ra = self.findings['regime_attribution']
            for regime, data in ra.items():
                if data.get('total_return', 0) < -0.05:
                    recommendations.append(f"3. **Review {data['name']} strategy** - This regime caused {data['pct_of_total_loss']:.0f}% of losses")
                    break
        
        for rec in recommendations:
            report += f"{rec}\n\n"
        
        report += """
---

## Next Steps

1. Run `v18_factor_calibration.py` to:
   - Remove factors with IC < -0.01
   - Flip factors with IC < -0.02 (strong negative signal is still signal)
   - Weight remaining factors by IC magnitude

2. Rerun `run_v17_full.py` with calibrated factors

3. Target: Move from -12.8% CAGR to **positive CAGR**
"""
        
        return report
    
    def run(self):
        """Run full diagnostic analysis"""
        print("\n" + "=" * 70)
        print("🔍 V18.0 DIAGNOSTIC ANALYSIS")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Run analyses
        self.analyze_regime_attribution()
        self.analyze_cost_vs_signal()
        self.calculate_factor_ic()
        self.validate_hmm_regime()
        
        # Generate report
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("✅ DIAGNOSTIC COMPLETE")
        print("=" * 70)
        print(f"\n📄 Report saved to: results/v18/DIAGNOSTIC_REPORT.md")
        
        return self.findings


def main():
    diagnostic = V18Diagnostic()
    findings = diagnostic.run()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 KEY FINDINGS SUMMARY")
    print("=" * 50)
    
    if 'cost_attribution' in findings:
        ca = findings['cost_attribution']
        print(f"\n💰 Cost Attribution:")
        print(f"   Loss from signals: {ca.get('pct_loss_from_signals', 0):.1f}%")
        print(f"   Loss from costs: {ca.get('pct_loss_from_costs', 0):.1f}%")
    
    if 'factor_ic' in findings:
        ic_df = findings['factor_ic']
        if isinstance(ic_df, pd.DataFrame):
            neg_count = (ic_df['mean_ic'] < 0).sum()
            print(f"\n📈 Factor IC:")
            print(f"   Negative IC factors: {neg_count}")
            print(f"   Positive IC factors: {len(ic_df) - neg_count}")
    
    if 'hmm_validation' in findings:
        hv = findings['hmm_validation']
        print(f"\n🧠 HMM Validation:")
        print(f"   Valid: {hv.get('hmm_is_valid', 'Unknown')}")
        if hv.get('issues'):
            for issue in hv['issues']:
                print(f"   ⚠️ {issue}")
    
    return findings


if __name__ == "__main__":
    main()
