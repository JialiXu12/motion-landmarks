"""
Script to read ribcage error RMSE from Excel file and calculate statistics
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

print(f"\nTotal rows in dataframe: {len(df)}")
print(f"\nColumn names in dataframe:")
print(df.columns.tolist())

# Check if ribcage error column exists
ribcage_col_candidates = [col for col in df.columns if 'ribcage' in col.lower() and 'rmse' in col.lower()]
print(f"\nRibcage RMSE column candidates: {ribcage_col_candidates}")

# Use the correct column name
if 'ribcage inlier RMSE' in df.columns:
    rmse_col = 'ribcage inlier RMSE'
elif len(ribcage_col_candidates) > 0:
    rmse_col = ribcage_col_candidates[0]
else:
    print("\nError: No ribcage RMSE column found!")
    print("Available columns containing 'ribcage' or 'error':")
    relevant_cols = [col for col in df.columns if 'ribcage' in col.lower() or 'error' in col.lower()]
    print(relevant_cols)
    exit()

print(f"\nUsing column: '{rmse_col}'")

# Get unique subjects and their RMSE values
# Each subject should have the same RMSE value across all their landmarks
subject_col = 'VL_ID'
if subject_col not in df.columns:
    print(f"\nError: Subject ID column '{subject_col}' not found!")
    exit()

# Group by subject and get the first RMSE value (should be the same for all rows of that subject)
subject_rmse = df.groupby(subject_col)[rmse_col].first().dropna()

print(f"\n{'=' * 80}")
print("RIBCAGE ALIGNMENT ERROR (RMSE) ANALYSIS")
print(f"{'=' * 80}")

print(f"\nNumber of unique subjects with RMSE data: {len(subject_rmse)}")
print(f"\nSubject IDs with RMSE data:")
print(sorted(subject_rmse.index.tolist()))

# Check for 64 subjects
if len(subject_rmse) != 64:
    print(f"\n⚠️ WARNING: Expected 64 subjects, but found {len(subject_rmse)} subjects with RMSE data")

    # Check how many subjects are in the dataset total
    total_subjects = df[subject_col].nunique()
    print(f"Total subjects in dataset: {total_subjects}")

    # Find subjects without RMSE data
    all_subjects = set(df[subject_col].unique())
    subjects_with_rmse = set(subject_rmse.index)
    subjects_without_rmse = all_subjects - subjects_with_rmse

    if subjects_without_rmse:
        print(f"\nSubjects without RMSE data ({len(subjects_without_rmse)}):")
        print(sorted(subjects_without_rmse))

# Calculate statistics
mean_rmse = subject_rmse.mean()
std_rmse = subject_rmse.std()
median_rmse = subject_rmse.median()
min_rmse = subject_rmse.min()
max_rmse = subject_rmse.max()

print(f"\n{'=' * 80}")
print("COHORT STATISTICS")
print(f"{'=' * 80}")
print(f"\nNumber of subjects (N): {len(subject_rmse)}")
print(f"Mean RMSE: {mean_rmse:.2f} mm")
print(f"Standard Deviation: {std_rmse:.2f} mm")
print(f"Median RMSE: {median_rmse:.2f} mm")
print(f"Range: {min_rmse:.2f} - {max_rmse:.2f} mm")

# Display individual subject values
print(f"\n{'=' * 80}")
print("INDIVIDUAL SUBJECT RMSE VALUES")
print(f"{'=' * 80}")
subject_rmse_sorted = subject_rmse.sort_index()
for subject_id, rmse_value in subject_rmse_sorted.items():
    print(f"Subject VL{subject_id:05d}: {rmse_value:.2f} mm")

print(f"\n{'=' * 80}")
print("SUMMARY FOR REPORTING")
print(f"{'=' * 80}")
print(f"\nRibcage alignment RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} mm (mean ± SD, N={len(subject_rmse)})")
print(f"Range: {min_rmse:.2f} to {max_rmse:.2f} mm")


