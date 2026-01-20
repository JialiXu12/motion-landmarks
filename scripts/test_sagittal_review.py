"""
Interactive test script to review and iterate on text visualization improvements
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import plot_sagittal_dual_axes, read_data

# Load the data using the same method as analysis.py
output_dir = Path("..") / "output"
data_file = output_dir / "landmark_results_v4_2026_01_12.xlsx"

print(f"Loading data from: {data_file}")
df_raw, df_ave, df_demo = read_data(data_file)

print(f"Data shape: {df_ave.shape}")
print(f"Number of records: {len(df_ave)}")

# Test with the 'breast' coloring scheme (default)
print("\n" + "="*80)
print("Generating plot with 'breast' coloring for review...")
print("="*80)
plot_sagittal_dual_axes(df_ave, color_by='breast')

# Display the saved plot
output_path = Path("..") / "output" / "figs" / "landmark vectors" / "Vectors_rel_sternum_sagittal_dual.png"
if output_path.exists():
    print(f"\n✅ Plot saved to: {output_path}")
    print("\nPlease review the plot and check:")
    print("  ✓ Text 'Right Breast' on left subplot is clearly visible")
    print("  ✓ Text 'Left Breast' on right subplot is clearly visible")
    print("  ✓ Both labels have white background boxes with colored borders")
    print("  ✓ Font size (13) is appropriate for readability")
    print("  ✓ Labels are positioned in upper portion without overlapping data")
    print("  ✓ Sentence case is used (not ALL CAPS)")
    print("  ✓ Overall presentation looks scientific and professional")

    # Try to display image info
    try:
        img = Image.open(output_path)
        print(f"\nImage dimensions: {img.size[0]} x {img.size[1]} pixels")
        print(f"Image mode: {img.mode}")
    except Exception as e:
        print(f"Could not read image details: {e}")
else:
    print(f"\n⚠️ Plot not found at: {output_path}")

print("\n" + "="*80)
print("Review complete. If improvements needed, iteration will continue...")
print("="*80)
