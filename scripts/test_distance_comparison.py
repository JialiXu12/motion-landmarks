"""
Test script for statistical comparison of DTS, DTN, and DTR
"""

import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from pathlib import Path

# Load data
OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v6_2026_02_10.xlsx"

print("Loading data...")
df_ave = pd.read_excel(EXCEL_FILE_PATH, sheet_name='processed_ave_data', engine='openpyxl')

print(f"Data loaded: {len(df_ave)} rows")
print(f"Columns: {df_ave.columns.tolist()}")

# Test the comparison for PRONE position
print("\n" + "=" * 80)
print("TESTING: DTS vs DTN vs DTR Comparison (PRONE)")
print("=" * 80)

# Find common indices (paired data)
common_idx_prone = df_ave[['Distance to skin (prone) [mm]',
                             'Distance to nipple (prone) [mm]',
                             'Distance to rib cage (prone) [mm]']].dropna().index

dts_prone = df_ave.loc[common_idx_prone, 'Distance to skin (prone) [mm]'].values
dtn_prone = df_ave.loc[common_idx_prone, 'Distance to nipple (prone) [mm]'].values
dtr_prone = df_ave.loc[common_idx_prone, 'Distance to rib cage (prone) [mm]'].values

n_prone = len(dts_prone)

print(f"\nSample size (paired): N = {n_prone}")
print(f"\nDescriptive Statistics:")
print(f"  DTS (Skin):     {dts_prone.mean():.2f} ± {dts_prone.std():.2f} mm")
print(f"  DTN (Nipple):   {dtn_prone.mean():.2f} ± {dtn_prone.std():.2f} mm")
print(f"  DTR (Rib Cage): {dtr_prone.mean():.2f} ± {dtr_prone.std():.2f} mm")

if n_prone >= 3:
    # Test normality
    print(f"\nNormality Tests (Shapiro-Wilk):")
    _, p_dts = stats.shapiro(dts_prone)
    _, p_dtn = stats.shapiro(dtn_prone)
    _, p_dtr = stats.shapiro(dtr_prone)

    print(f"  DTS: p = {p_dts:.4f} {'(Normal)' if p_dts > 0.05 else '(Non-normal)'}")
    print(f"  DTN: p = {p_dtn:.4f} {'(Normal)' if p_dtn > 0.05 else '(Non-normal)'}")
    print(f"  DTR: p = {p_dtr:.4f} {'(Normal)' if p_dtr > 0.05 else '(Non-normal)'}")

    all_normal = (p_dts > 0.05) and (p_dtn > 0.05) and (p_dtr > 0.05)

    if all_normal:
        print(f"\n→ Using Repeated Measures ANOVA")

        # Prepare data
        data_long = pd.DataFrame({
            'subject': np.tile(np.arange(n_prone), 3),
            'metric': np.repeat(['DTS', 'DTN', 'DTR'], n_prone),
            'distance': np.concatenate([dts_prone, dtn_prone, dtr_prone])
        })

        # Run RM-ANOVA
        rm_results = pg.rm_anova(data=data_long, dv='distance', within='metric', subject='subject')

        print(f"\nResults:")
        print(f"  F = {rm_results['F'].values[0]:.3f}")
        print(f"  p-value = {rm_results['p-unc'].values[0]:.4e}")
        print(f"  Effect size (η²) = {rm_results['np2'].values[0]:.3f}")

        if rm_results['p-unc'].values[0] < 0.05:
            print(f"\n  ✓ Significant difference detected!")

            # Post-hoc
            print(f"\nPost-hoc Pairwise Comparisons:")
            posthoc = pg.pairwise_ttests(data=data_long, dv='distance', within='metric',
                                        subject='subject', padjust='bonf')
            print(posthoc[['A', 'B', 'T', 'p-unc', 'p-corr']])
    else:
        print(f"\n→ Using Friedman Test (non-parametric)")

        friedman_stat, p_friedman = stats.friedmanchisquare(dts_prone, dtn_prone, dtr_prone)

        print(f"\nResults:")
        print(f"  χ²(2) = {friedman_stat:.3f}")
        print(f"  p-value = {p_friedman:.4e}")

        if p_friedman < 0.05:
            print(f"\n  ✓ Significant difference detected!")

            # Post-hoc Wilcoxon
            print(f"\nPost-hoc Pairwise Comparisons (Wilcoxon + Bonferroni):")
            comparisons = [
                ('DTS vs DTN', dts_prone, dtn_prone),
                ('DTS vs DTR', dts_prone, dtr_prone),
                ('DTN vs DTR', dtn_prone, dtr_prone)
            ]

            for name, d1, d2 in comparisons:
                _, p_wilcox = stats.wilcoxon(d1, d2)
                p_corr = min(p_wilcox * 3, 1.0)
                sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
                print(f"  {name:15s}: p = {p_corr:.4e} ({sig})")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

