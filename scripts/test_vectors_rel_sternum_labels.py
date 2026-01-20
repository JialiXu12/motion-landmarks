"""
Test script to verify the changes to plot_vectors_rel_sternum
"""
from pathlib import Path
from analysis import plot_vectors_rel_sternum, read_data

print("="*80)
print("TEST: plot_vectors_rel_sternum Text Label Updates")
print("="*80)

# Load data
data_file = Path("../output/landmark_results_v4_2026_01_12.xlsx")
print(f"\nLoading data from: {data_file}")
df_raw, df_ave, df_demo = read_data(data_file)
print(f"✓ Data loaded: {len(df_ave)} records\n")

print("="*80)
print("TESTING CHANGES")
print("="*80)

print("\n1. Testing color_by='breast' (default)")
print("   Expected:")
print("   - Axial: Blue 'Right Breast', Green 'Left Breast'")
print("   - Coronal: Blue 'Right Breast', Green 'Left Breast'")
print("   - Sagittal: Blue 'Right Breast' (moved right), Green 'Left Breast' (moved left)")
plot_vectors_rel_sternum(df_ave, color_by='breast')

print("\n2. Testing color_by='dts'")
print("   Expected:")
print("   - Axial: Black 'Right Breast', Black 'Left Breast'")
print("   - Coronal: Black 'Right Breast', Black 'Left Breast'")
print("   - Sagittal: NO text labels")
plot_vectors_rel_sternum(df_ave, color_by='dts')

print("\n3. Testing color_by='subject'")
print("   Expected:")
print("   - Axial: Black 'Right Breast', Black 'Left Breast'")
print("   - Coronal: Black 'Right Breast', Black 'Left Breast'")
print("   - Sagittal: NO text labels")
plot_vectors_rel_sternum(df_ave, color_by='subject')

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)
print("\nPlease verify the generated plots:")
print("  1. Check ../output/figs/landmark vectors/ for:")
print("     - Vectors_rel_sternum_Axial.png")
print("     - Vectors_rel_sternum_Coronal.png")
print("     - Vectors_rel_sternum_Sagittal.png")
print("     - Vectors_rel_sternum_Axial_DTS.png")
print("     - Vectors_rel_sternum_Coronal_DTS.png")
print("     - Vectors_rel_sternum_Sagittal_DTS.png")
print("     - Vectors_rel_sternum_Axial_by_subject.png")
print("     - Vectors_rel_sternum_Coronal_by_subject.png")
print("     - Vectors_rel_sternum_Sagittal_by_subject.png")
print("\n  2. Verify text labels match the expected behavior above")
