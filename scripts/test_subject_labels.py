"""
Quick test to verify breast labels appear when color_by='subject'
"""
from pathlib import Path
from analysis import plot_sagittal_dual_axes, read_data

print("="*80)
print("TEST: Breast Labels Visibility with Subject Coloring")
print("="*80)

# Load data
data_file = Path("../output/landmark_results_v4_2026_01_12.xlsx")
print(f"\nLoading data from: {data_file}")
df_raw, df_ave, df_demo = read_data(data_file)
print(f"✓ Data loaded: {len(df_ave)} records\n")

# Test with subject coloring
print("Generating plot with color_by='subject'...")
print("Expected: Both 'Right Breast' and 'Left Breast' labels should be visible")
print("          Labels should have black text with white background boxes\n")

plot_sagittal_dual_axes(df_ave, color_by='subject')

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)
print("\nPlease verify:")
print("  1. Open: ../output/figs/landmark vectors/Vectors_rel_sternum_sagittal_dual_by_subject.png")
print("  2. Check: 'Right Breast' label visible on left subplot")
print("  3. Check: 'Left Breast' label visible on right subplot")
print("  4. Both labels should have:")
print("     - Black text (since color_by != 'breast')")
print("     - White background box with black border")
print("     - Font size 14, bold")
print("     - Positioned at upper portion (82% of y-axis)")
print("\nIf all checks pass, the issue is RESOLVED! ✅")
