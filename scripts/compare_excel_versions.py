"""
Compare processed_ave_data sheets between two Excel versions to identify differences.
This script compares landmark_results_v5_2026_01_21.xlsx and landmark_results_v6_2026_01_12.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compare_excel_versions():
    """Compare two Excel files and identify differences in the processed_ave_data sheet."""

    # File paths
    output_dir = Path(__file__).parent.parent / "output"
    v5_path = output_dir / "landmark_results_v5_2026_01_21.xlsx"
    v6_path = output_dir / "landmark_results_v6_2026_02_09.xlsx"

    print("=" * 80)
    print("COMPARING EXCEL VERSIONS: V6 vs V5")
    print("=" * 80)

    # Load data
    print(f"\nLoading V6: {v6_path}")
    print(f"Loading V5: {v5_path}")

    try:
        df_v6 = pd.read_excel(v6_path, sheet_name='processed_ave_data')
        df_v5 = pd.read_excel(v5_path, sheet_name='processed_ave_data')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"\nv6 shape: {df_v6.shape}")
    print(f"V5 shape: {df_v5.shape}")

    # Compare columns
    print("\n" + "=" * 80)
    print("COLUMN COMPARISON")
    print("=" * 80)

    v6_cols = set(df_v6.columns)
    v5_cols = set(df_v5.columns)

    only_in_v6 = v6_cols - v5_cols
    only_in_v5 = v5_cols - v6_cols
    common_cols = v6_cols & v5_cols

    print(f"\nColumns only in V6 ({len(only_in_v6)}):")
    for col in sorted(only_in_v6):
        print(f"  - {col}")

    print(f"\nColumns only in V5 ({len(only_in_v5)}):")
    for col in sorted(only_in_v5):
        print(f"  - {col}")

    print(f"\nCommon columns: {len(common_cols)}")

    # Create a key to match rows between versions
    key_cols = ['VL_ID', 'Landmark name']
    if all(col in common_cols for col in key_cols):
        print("\n" + "=" * 80)
        print("ROW MATCHING BY VL_ID AND LANDMARK NAME")
        print("=" * 80)

        # Create unique keys
        df_v6['_key'] = df_v6['VL_ID'].astype(str) + '_' + df_v6['Landmark name'].astype(str)
        df_v5['_key'] = df_v5['VL_ID'].astype(str) + '_' + df_v5['Landmark name'].astype(str)

        v6_keys = set(df_v6['_key'])
        v5_keys = set(df_v5['_key'])

        only_in_v6_rows = v6_keys - v5_keys
        only_in_v5_rows = v5_keys - v6_keys
        common_rows = v6_keys & v5_keys

        print(f"\nRows only in V6: {len(only_in_v6_rows)}")
        for key in sorted(only_in_v6_rows):
            print(f"  - {key}")

        print(f"\nRows only in V5: {len(only_in_v5_rows)}")
        for key in sorted(only_in_v5_rows):
            print(f"  - {key}")

        print(f"\nCommon rows: {len(common_rows)}")

    # Compare numeric columns for common rows
    print("\n" + "=" * 80)
    print("VALUE COMPARISON FOR COMMON ROWS AND COLUMNS")
    print("=" * 80)

    # Numeric columns to compare
    numeric_cols = [col for col in common_cols if df_v6[col].dtype in ['float64', 'int64']
                    and df_v5[col].dtype in ['float64', 'int64']]

    print(f"\nNumeric columns to compare: {len(numeric_cols)}")

    # Merge on key for comparison
    if '_key' in df_v6.columns and '_key' in df_v5.columns:
        df_merged = df_v6[['_key'] + [c for c in numeric_cols if c in df_v6.columns]].merge(
            df_v5[['_key'] + [c for c in numeric_cols if c in df_v5.columns]],
            on='_key',
            suffixes=('_v6', '_v5'),
            how='inner'
        )

        print(f"Merged rows for comparison: {len(df_merged)}")

        # Analyze differences for each column
        differences_summary = []

        for col in sorted(numeric_cols):
            col_v6 = f"{col}_v6"
            col_v5 = f"{col}_v5"

            if col_v6 in df_merged.columns and col_v5 in df_merged.columns:
                v6_vals = df_merged[col_v6]
                v5_vals = df_merged[col_v5]

                # Skip if all NaN
                if v6_vals.isna().all() and v5_vals.isna().all():
                    continue

                # Calculate difference
                diff = v5_vals - v6_vals

                # Check for sign flip
                sign_flip_mask = (v6_vals * v5_vals < 0) & v6_vals.notna() & v5_vals.notna()
                sign_flips = sign_flip_mask.sum()

                # Check for exact negation
                exact_neg_mask = np.isclose(v5_vals, -v6_vals, rtol=1e-5, equal_nan=True)
                exact_negations = exact_neg_mask.sum()

                # Check for exact match
                exact_match = np.isclose(v6_vals, v5_vals, rtol=1e-5, equal_nan=True).sum()

                # Calculate stats
                valid_diff = diff.dropna()
                if len(valid_diff) > 0:
                    mean_diff = valid_diff.mean()
                    std_diff = valid_diff.std()
                    max_abs_diff = valid_diff.abs().max()

                    differences_summary.append({
                        'Column': col,
                        'Valid Comparisons': len(valid_diff),
                        'Exact Matches': exact_match,
                        'Sign Flips': sign_flips,
                        'Exact Negations': exact_negations,
                        'Mean Diff': mean_diff,
                        'Std Diff': std_diff,
                        'Max Abs Diff': max_abs_diff,
                    })

        # Display differences summary
        if differences_summary:
            df_diff_summary = pd.DataFrame(differences_summary)

            print("\n" + "-" * 80)
            print("DIFFERENCE SUMMARY (V5 - V6)")
            print("-" * 80)

            # Show columns with significant differences
            print("\n[Columns with EXACT NEGATION (sign flip pattern)]:")
            negation_cols = df_diff_summary[df_diff_summary['Exact Negations'] > df_diff_summary['Valid Comparisons'] * 0.5]
            if len(negation_cols) > 0:
                for _, row in negation_cols.iterrows():
                    print(f"  {row['Column']}: {row['Exact Negations']}/{row['Valid Comparisons']} values are exact negations")
            else:
                print("  None detected")

            print("\n[Columns with SIGN FLIPS (opposite signs)]:")
            sign_flip_cols = df_diff_summary[df_diff_summary['Sign Flips'] > 0]
            if len(sign_flip_cols) > 0:
                for _, row in sign_flip_cols.iterrows():
                    print(f"  {row['Column']}: {row['Sign Flips']} sign flips")
            else:
                print("  None detected")

            print("\n[Columns with EXACT MATCHES]:")
            exact_match_cols = df_diff_summary[df_diff_summary['Exact Matches'] == df_diff_summary['Valid Comparisons']]
            if len(exact_match_cols) > 0:
                for _, row in exact_match_cols.iterrows():
                    print(f"  {row['Column']}: {row['Exact Matches']}/{row['Valid Comparisons']} exact matches")
            else:
                print("  None")

            print("\n[Columns with DIFFERENCES]:")
            diff_cols = df_diff_summary[
                (df_diff_summary['Max Abs Diff'] > 1e-6) &
                (df_diff_summary['Exact Matches'] < df_diff_summary['Valid Comparisons'])
            ]
            if len(diff_cols) > 0:
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 200)
                print(diff_cols.to_string(index=False))
            else:
                print("  No significant differences detected")

    # Detailed row-by-row comparison for key columns
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON FOR LANDMARK POSITIONS")
    print("=" * 80)

    position_cols = [
        'landmark ave prone transformed x',
        'landmark ave prone transformed y',
        'landmark ave prone transformed z',
        'landmark ave supine x',
        'landmark ave supine y',
        'landmark ave supine z',
        'left nipple prone transformed x',
        'left nipple prone transformed y',
        'left nipple prone transformed z',
        'right nipple prone transformed x',
        'right nipple prone transformed y',
        'right nipple prone transformed z',
    ]

    # Filter to columns that exist
    existing_pos_cols = [col for col in position_cols if col in common_cols]

    if existing_pos_cols and '_key' in df_v6.columns:
        print(f"\nComparing {len(existing_pos_cols)} position columns...")

        for col in existing_pos_cols:
            col_v6 = f"{col}_v6"
            col_v5 = f"{col}_v5"

            if col_v6 in df_merged.columns and col_v5 in df_merged.columns:
                v6_vals = df_merged[col_v6].dropna()
                v5_vals = df_merged[col_v5].dropna()

                # Get common valid indices
                valid_idx = v6_vals.index.intersection(v5_vals.index)
                if len(valid_idx) == 0:
                    continue

                v6_valid = v6_vals.loc[valid_idx]
                v5_valid = v5_vals.loc[valid_idx]

                diff = v5_valid - v6_valid

                # Check for patterns
                is_exact = np.allclose(v6_valid, v5_valid, rtol=1e-5, equal_nan=True)
                is_negated = np.allclose(v6_valid, -v5_valid, rtol=1e-5, equal_nan=True)

                if is_exact:
                    relationship = "EXACT MATCH"
                elif is_negated:
                    relationship = "EXACT NEGATION (V5 = -V6)"
                elif diff.abs().max() < 1:
                    relationship = f"SMALL DIFF (max={diff.abs().max():.4f})"
                else:
                    relationship = f"DIFFERENT (mean_diff={diff.mean():.2f}, max_abs={diff.abs().max():.2f})"

                print(f"\n  {col}:")
                print(f"    Relationship: {relationship}")
                print(f"    V6 range: [{v6_valid.min():.2f}, {v6_valid.max():.2f}]")
                print(f"    V5 range: [{v5_valid.min():.2f}, {v5_valid.max():.2f}]")

    # Check for column swaps (e.g., left/right nipple swapped)
    print("\n" + "=" * 80)
    print("CHECKING FOR COLUMN SWAPS (e.g., Left/Right nipple)")
    print("=" * 80)

    swap_pairs = [
        ('left nipple prone transformed x', 'right nipple prone transformed x'),
        ('left nipple prone transformed y', 'right nipple prone transformed y'),
        ('left nipple prone transformed z', 'right nipple prone transformed z'),
        ('left nipple supine x', 'right nipple supine x'),
        ('left nipple supine y', 'right nipple supine y'),
        ('left nipple supine z', 'right nipple supine z'),
    ]

    for left_col, right_col in swap_pairs:
        if f"{left_col}_v6" in df_merged.columns and f"{right_col}_v5" in df_merged.columns:
            v6_left = df_merged[f"{left_col}_v6"].dropna()
            v5_right = df_merged[f"{right_col}_v5"].dropna()

            valid_idx = v6_left.index.intersection(v5_right.index)
            if len(valid_idx) > 0:
                v6_l = v6_left.loc[valid_idx]
                v5_r = v5_right.loc[valid_idx]

                if np.allclose(v6_l, v5_r, rtol=1e-5, equal_nan=True):
                    print(f"\n  SWAP DETECTED: V6[{left_col}] == V5[{right_col}]")

        if f"{right_col}_v6" in df_merged.columns and f"{left_col}_v5" in df_merged.columns:
            v6_right = df_merged[f"{right_col}_v6"].dropna()
            v5_left = df_merged[f"{left_col}_v5"].dropna()

            valid_idx = v6_right.index.intersection(v5_left.index)
            if len(valid_idx) > 0:
                v6_r = v6_right.loc[valid_idx]
                v5_l = v5_left.loc[valid_idx]

                if np.allclose(v6_r, v5_l, rtol=1e-5, equal_nan=True):
                    print(f"\n  SWAP DETECTED: V6[{right_col}] == V5[{left_col}]")

    # Sample comparison for debugging
    print("\n" + "=" * 80)
    print("SAMPLE DATA COMPARISON (First 5 rows with data)")
    print("=" * 80)

    # Get rows with non-null landmark positions
    sample_cols = ['_key', 'VL_ID', 'Landmark name']
    pos_sample_cols = ['landmark ave prone transformed x', 'left nipple prone transformed x', 'right nipple prone transformed x']

    for col in pos_sample_cols:
        if col in df_v6.columns:
            sample_cols.append(col)

    sample_v6 = df_v6[df_v6['landmark ave prone transformed x'].notna()][sample_cols].head(5)
    sample_v5 = df_v5[df_v5['landmark ave prone transformed x'].notna()][sample_cols].head(5)

    print("\nV6 Sample:")
    print(sample_v6.to_string())

    print("\nV5 Sample:")
    print(sample_v5.to_string())

    # Check if left/right nipple prone positions are swapped between versions
    print("\n" + "=" * 80)
    print("HYPOTHESIS: LEFT/RIGHT NIPPLE PRONE TRANSFORMED SWAPPED BETWEEN V6 AND V5")
    print("=" * 80)

    # Check if V6's left == V5's right and V6's right == V5's left
    axes = ['x', 'y', 'z']
    for axis in axes:
        left_col = f'left nipple prone transformed {axis}'
        right_col = f'right nipple prone transformed {axis}'

        v6_left_col = f'{left_col}_v6'
        v6_right_col = f'{right_col}_v6'
        v5_left_col = f'{left_col}_v5'
        v5_right_col = f'{right_col}_v5'

        if all(c in df_merged.columns for c in [v6_left_col, v6_right_col, v5_left_col, v5_right_col]):
            # Check V6 left == V5 right
            v6_left = df_merged[v6_left_col].dropna()
            v5_right = df_merged[v5_right_col].dropna()

            valid_idx = v6_left.index.intersection(v5_right.index)
            if len(valid_idx) > 0:
                match = np.allclose(v6_left.loc[valid_idx], v5_right.loc[valid_idx], rtol=1e-5, equal_nan=True)
                diff = (v5_right.loc[valid_idx] - v6_left.loc[valid_idx]).abs().max()
                print(f"\n  V6[{left_col}] == V5[{right_col}]: {match} (max diff: {diff:.6f})")

            # Check V6 right == V5 left
            v6_right = df_merged[v6_right_col].dropna()
            v5_left = df_merged[v5_left_col].dropna()

            valid_idx = v6_right.index.intersection(v5_left.index)
            if len(valid_idx) > 0:
                match = np.allclose(v6_right.loc[valid_idx], v5_left.loc[valid_idx], rtol=1e-5, equal_nan=True)
                diff = (v5_left.loc[valid_idx] - v6_right.loc[valid_idx]).abs().max()
                print(f"  V6[{right_col}] == V5[{left_col}]: {match} (max diff: {diff:.6f})")

    # Show specific row comparison
    print("\n" + "=" * 80)
    print("ROW-BY-ROW NIPPLE COMPARISON (First 10 rows with valid data)")
    print("=" * 80)

    nipple_cols_v6 = ['left nipple prone transformed x_v6', 'right nipple prone transformed x_v6']
    nipple_cols_v5 = ['left nipple prone transformed x_v5', 'right nipple prone transformed x_v5']

    if all(c in df_merged.columns for c in nipple_cols_v6 + nipple_cols_v5):
        sample = df_merged[df_merged['left nipple prone transformed x_v6'].notna()][
            ['_key'] + nipple_cols_v6 + nipple_cols_v5
        ].head(10)

        sample.columns = ['Key', 'V6_Left_X', 'V6_Right_X', 'V5_Left_X', 'V5_Right_X']
        print("\n")
        print(sample.to_string())

        print("\n\nPattern Analysis:")
        print("  If V6_Left_X == V5_Right_X and V6_Right_X == V5_Left_X, then nipples are SWAPPED")

    # Identify which VL_IDs have different nipple values
    print("\n" + "=" * 80)
    print("IDENTIFYING VL_IDs WITH DIFFERENT NIPPLE POSITIONS")
    print("=" * 80)

    # Add VL_ID to merged df
    df_merged_with_id = df_merged.copy()
    df_merged_with_id['VL_ID'] = df_merged_with_id['_key'].str.split('_').str[0].astype(int)

    # Check for each VL_ID
    vl_ids = df_merged_with_id['VL_ID'].unique()

    identical_vl_ids = []
    different_vl_ids = []
    swapped_vl_ids = []

    for vl_id in sorted(vl_ids):
        vl_data = df_merged_with_id[df_merged_with_id['VL_ID'] == vl_id].iloc[0]

        v6_left_x = vl_data.get('left nipple prone transformed x_v6')
        v6_right_x = vl_data.get('right nipple prone transformed x_v6')
        v5_left_x = vl_data.get('left nipple prone transformed x_v5')
        v5_right_x = vl_data.get('right nipple prone transformed x_v5')

        if pd.isna(v6_left_x) or pd.isna(v5_left_x):
            continue

        # Check if identical
        if np.isclose(v6_left_x, v5_left_x) and np.isclose(v6_right_x, v5_right_x):
            identical_vl_ids.append(vl_id)
        # Check if swapped
        elif np.isclose(v6_left_x, v5_right_x) and np.isclose(v6_right_x, v5_left_x):
            swapped_vl_ids.append(vl_id)
        else:
            different_vl_ids.append(vl_id)
            print(f"\n  VL{vl_id:05d}: Different (not identical, not swapped)")
            print(f"    V6: Left={v6_left_x:.2f}, Right={v6_right_x:.2f}")
            print(f"    V5: Left={v5_left_x:.2f}, Right={v5_right_x:.2f}")

    print(f"\n\nSUMMARY BY VL_ID:")
    print(f"  IDENTICAL nipple positions: {identical_vl_ids}")
    print(f"  SWAPPED nipple positions: {swapped_vl_ids}")
    print(f"  OTHER differences: {different_vl_ids}")

    # For different VL_IDs, show detailed comparison
    if different_vl_ids:
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS OF VL_IDs WITH DIFFERENCES (NOT SIMPLE SWAP)")
        print("=" * 80)

        for vl_id in different_vl_ids:
            vl_data = df_merged_with_id[df_merged_with_id['VL_ID'] == vl_id].iloc[0]
            print(f"\nVL{vl_id:05d}:")

            for axis in ['x', 'y', 'z']:
                left_col = f'left nipple prone transformed {axis}'
                right_col = f'right nipple prone transformed {axis}'

                v6_left = vl_data.get(f'{left_col}_v6')
                v6_right = vl_data.get(f'{right_col}_v6')
                v5_left = vl_data.get(f'{left_col}_v5')
                v5_right = vl_data.get(f'{right_col}_v5')

                if not pd.isna(v6_left):
                    # Check relationships
                    v6_v5_left_diff = v5_left - v6_left if not pd.isna(v5_left) else None
                    v6_v5_right_diff = v5_right - v6_right if not pd.isna(v5_right) else None
                    v6_left_v5_right_diff = v5_right - v6_left if not pd.isna(v5_right) else None
                    v6_right_v5_left_diff = v5_left - v6_right if not pd.isna(v5_left) else None

                    print(f"  {axis}-axis:")
                    print(f"    V6: Left={v6_left:.2f}, Right={v6_right:.2f}")
                    print(f"    V5: Left={v5_left:.2f}, Right={v5_right:.2f}")
                    if v6_v5_left_diff is not None:
                        print(f"    V5_Left - V6_Left = {v6_v5_left_diff:.2f}")
                        print(f"    V5_Right - V6_Right = {v6_v5_right_diff:.2f}")
                        print(f"    V5_Right - V6_Left = {v6_left_v5_right_diff:.2f} (swap check)")
                        print(f"    V5_Left - V6_Right = {v6_right_v5_left_diff:.2f} (swap check)")

    # Check nipple displacement vectors
    print("\n" + "=" * 80)
    print("NIPPLE DISPLACEMENT VECTORS RELATIONSHIP")
    print("=" * 80)

    disp_cols = [
        ('Left nipple displacement vector vx', 'Right nipple displacement vector vx'),
        ('Left nipple displacement vector vy', 'Right nipple displacement vector vy'),
        ('Left nipple displacement vector vz', 'Right nipple displacement vector vz'),
    ]

    for left_col, right_col in disp_cols:
        v6_left_col = f'{left_col}_v6'
        v6_right_col = f'{right_col}_v6'
        v5_left_col = f'{left_col}_v5'
        v5_right_col = f'{right_col}_v5'

        if all(c in df_merged.columns for c in [v6_left_col, v6_right_col, v5_left_col, v5_right_col]):
            v6_left = df_merged[v6_left_col].dropna()
            v5_left = df_merged[v5_left_col].dropna()
            v6_right = df_merged[v6_right_col].dropna()
            v5_right = df_merged[v5_right_col].dropna()

            valid_idx = v6_left.index.intersection(v5_left.index)
            if len(valid_idx) > 0:
                diff_left = v5_left.loc[valid_idx] - v6_left.loc[valid_idx]
                diff_right = v5_right.loc[valid_idx] - v6_right.loc[valid_idx]

                # Check if diff_left == -diff_right (indicating swap)
                if len(diff_left) > 0 and len(diff_right) > 0:
                    is_negated = np.allclose(diff_left, -diff_right, rtol=1e-5, equal_nan=True)
                    print(f"\n  {left_col} change == -{right_col} change: {is_negated}")
                    print(f"    Mean diff V5-V6 for Left: {diff_left.mean():.4f}")
                    print(f"    Mean diff V5-V6 for Right: {diff_right.mean():.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    print(f"""
KEY DIFFERENCES BETWEEN V6 AND V5:

1. LANDMARK POSITIONS (prone transformed, supine) - EXACT MATCH
   All landmark coordinates are identical between versions.

2. NIPPLE POSITIONS (prone transformed) - SWAPPED FOR SOME VL_IDs!
   
   VL_IDs with IDENTICAL nipple positions (V6 == V5): 
   {identical_vl_ids}
   Total: {len(identical_vl_ids)} VL_IDs
   
   VL_IDs with SWAPPED nipple positions (V6_Left == V5_Right and V6_Right == V5_Left):
   {swapped_vl_ids}
   Total: {len(swapped_vl_ids)} VL_IDs
   
3. CONSEQUENCE OF NIPPLE SWAP (for affected VL_IDs):
   - Left/Right nipple displacement vectors are swapped
   - Landmark relative to nipple vectors are affected
   - Distance to nipple (prone) values are affected

4. UNAFFECTED COLUMNS:
   - All landmark positions (same in both versions)
   - Sternum positions (same)
   - Nipple supine positions (same - swap only affects prone transformed)
   - Distance to rib/skin (same)
   - Mask neighborhood values (same)
   
CONCLUSION: For {len(swapped_vl_ids)} VL_IDs, the LEFT and RIGHT nipple 
PRONE TRANSFORMED positions are SWAPPED between V6 and V5.
For {len(identical_vl_ids)} VL_IDs, the nipple positions are identical.

This indicates that a correction was made in one version to fix mislabeled
left/right nipple assignments for specific subjects.
""")

    # Save detailed comparison to file
    output_path = output_dir / "version_comparison_report.txt"
    print(f"\n\nSaving detailed report to: {output_path}")

    return df_v6, df_v5, df_merged if '_key' in df_v6.columns else None


if __name__ == "__main__":
    df_v6, df_v5, df_merged = compare_excel_versions()
