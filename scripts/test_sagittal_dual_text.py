"""
Test script for improving text visualization in plot_sagittal_dual_axes
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import plot_sagittal_dual_axes, read_data

# Load the data using the same method as analysis.py
output_dir = Path("..") / "output"
data_file = output_dir / "landmark_results_v4_2026_01_12.xlsx"

print(f"Loading data from: {data_file}")
df_raw, df_ave, df_demo = read_data(data_file)

print(f"Data shape: {df_ave.shape}")
print(f"Columns: {df_ave.columns.tolist()[:15]}...")
print(f"Number of records: {len(df_ave)}")

# Test with different coloring schemes
print("\n" + "="*80)
print("Testing plot_sagittal_dual_axes with default 'breast' coloring...")
print("="*80)
plot_sagittal_dual_axes(df_ave, color_by='breast')

print("\n" + "="*80)
print("Testing plot_sagittal_dual_axes with 'dts' coloring...")
print("="*80)
plot_sagittal_dual_axes(df_ave, color_by='dts')

print("\n" + "="*80)
print("Testing plot_sagittal_dual_axes with 'subject' coloring...")
print("="*80)
plot_sagittal_dual_axes(df_ave, color_by='subject')

print("\n✅ All tests completed!")
print("\n" + "="*80)
print("Please review the generated plots to check:")
print("  1. Text labels are readable and well-positioned")
print("  2. Text has appropriate background/border for visibility")
print("  3. Font size is appropriate for scientific presentation")
print("  4. Labels use sentence case (e.g., 'Right Breast' not 'RIGHT BREAST')")
print("  5. Overall appearance is professional and publication-ready")
print("="*80)
