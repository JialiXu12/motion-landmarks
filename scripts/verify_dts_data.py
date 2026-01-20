"""
Quick verification script to check DTS plotting
"""

from pathlib import Path
from analysis import read_data
import numpy as np

# Load data
df_raw, df_ave, df_demo = read_data(Path('../output/landmark_results_v4_2026_01_12.xlsx'))

# Separate by breast
left_df = df_ave[df_ave['landmark side (prone)'] == 'LB']
right_df = df_ave[df_ave['landmark side (prone)'] == 'RB']

print("=" * 80)
print("DATA VERIFICATION FOR DTS PLOTTING")
print("=" * 80)

print(f"\nTotal landmarks in dataset: {len(df_ave)}")
print(f"Right breast landmarks: {len(right_df)}")
print(f"Left breast landmarks: {len(left_df)}")

# Check DTS values
dts_right = right_df['Distance to skin (prone) [mm]'].values
dts_left = left_df['Distance to skin (prone) [mm]'].values

print(f"\nRight breast DTS values:")
print(f"  - Count: {len(dts_right)}")
print(f"  - Min: {np.min(dts_right):.2f} mm")
print(f"  - Max: {np.max(dts_right):.2f} mm")
print(f"  - Mean: {np.mean(dts_right):.2f} mm")
print(f"  - Null values: {np.isnan(dts_right).sum()}")

print(f"\nLeft breast DTS values:")
print(f"  - Count: {len(dts_left)}")
print(f"  - Min: {np.min(dts_left):.2f} mm")
print(f"  - Max: {np.max(dts_left):.2f} mm")
print(f"  - Mean: {np.mean(dts_left):.2f} mm")
print(f"  - Null values: {np.isnan(dts_left).sum()}")

# Check vector magnitudes
prone_x_r = right_df['landmark ave prone transformed x'].values
prone_y_r = right_df['landmark ave prone transformed y'].values
prone_z_r = right_df['landmark ave prone transformed z'].values
supine_x_r = right_df['landmark ave supine x'].values
supine_y_r = right_df['landmark ave supine y'].values
supine_z_r = right_df['landmark ave supine z'].values

vec_x_r = supine_x_r - prone_x_r
vec_y_r = supine_y_r - prone_y_r
vec_z_r = supine_z_r - prone_z_r

magnitudes_r = np.sqrt(vec_x_r**2 + vec_y_r**2 + vec_z_r**2)

print(f"\nRight breast vector magnitudes:")
print(f"  - Min: {np.min(magnitudes_r):.2f} mm")
print(f"  - Max: {np.max(magnitudes_r):.2f} mm")
print(f"  - Mean: {np.mean(magnitudes_r):.2f} mm")

print("\n" + "=" * 80)
print("All landmarks should be plotted!")
print("=" * 80)
