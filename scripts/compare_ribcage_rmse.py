"""
Script to compare ribcage error RMSE columns and calculate statistics
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Define paths
OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v6_2026_02_10.xlsx"

# Read the Excel file
print(f"Reading data from: {EXCEL_FILE_PATH}")
df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='processed_ave_data', engine='openpyxl')

subject_col = 'VL_ID'

# Check both RMSE columns
rmse_cols = ['ribcage error rmse', 'ribcage inlier rmse']

print(f"\n{'=' * 80}")
print("COMPARISON: RIBCAGE ERROR RMSE vs INLIER RMSE")
print(f"{'=' * 80}")

for rmse_col in rmse_cols:
    if rmse_col in df.columns:
        # Get unique subjects and their RMSE values
        subject_rmse = df.groupby(subject_col)[rmse_col].first().dropna()

        # Calculate statistics
        mean_rmse = subject_rmse.mean()
        std_rmse = subject_rmse.std()
        median_rmse = subject_rmse.median()
        min_rmse = subject_rmse.min()
        max_rmse = subject_rmse.max()

        print(f"\n{rmse_col.upper()}")
        print(f"{'-' * 80}")
        print(f"Number of subjects: {len(subject_rmse)}")
        print(f"Mean: {mean_rmse:.2f} mm")
        print(f"Standard Deviation: {std_rmse:.2f} mm")
        print(f"Median: {median_rmse:.2f} mm")
        print(f"Range: {min_rmse:.2f} - {max_rmse:.2f} mm")

print(f"\n{'=' * 80}")
print("RECOMMENDED FOR REPORTING")
print(f"{'=' * 80}")

# Use ribcage inlier rmse if available (more accurate as it excludes outliers)
if 'ribcage inlier rmse' in df.columns:
    rmse_col = 'ribcage inlier rmse'
    print("\nUsing 'ribcage inlier rmse' (recommended - excludes outlier points)")
else:
    rmse_col = 'ribcage error rmse'
    print("\nUsing 'ribcage error rmse'")

subject_rmse = df.groupby(subject_col)[rmse_col].first().dropna()
mean_rmse = subject_rmse.mean()
std_rmse = subject_rmse.std()
min_rmse = subject_rmse.min()
max_rmse = subject_rmse.max()

print(f"\n**Ribcage alignment accuracy: {mean_rmse:.2f} ± {std_rmse:.2f} mm (mean ± SD, N={len(subject_rmse)})**")
print(f"Range: {min_rmse:.2f} to {max_rmse:.2f} mm")

print(f"\n{'=' * 80}")
print("SUBJECTS IN DATASET")
print(f"{'=' * 80}")
print(f"Total subjects with RMSE data: {len(subject_rmse)}")
print(f"Note: Found {len(subject_rmse)} subjects (expected 64 - may have 1 missing)")


