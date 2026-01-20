"""
Example script demonstrating the improved plot_nipple_relative_from_sternum_data function.

The function now accepts a DataFrame directly and reads nipple coordinates from
DataFrame columns (left/right nipple prone/supine transformed x/y/z) instead of
requiring manual data extraction.
"""

from analysis import read_data, plot_nipple_relative_landmarks

# Define your Excel file path
EXCEL_FILE_PATH = r"../output/landmark_results_v4_2026_01_12.xlsx"

def main():
    # Load data
    print("Loading data...")
    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)

    print(f"\nLoaded {len(df_ave)} landmark records")

    # Example 1: Plot for a specific subject (e.g., VL 81)
    print("\n" + "="*70)
    print("Example 1: Plotting nipple-relative motion for Subject VL 81")
    print("="*70)

    plot_nipple_relative_landmarks(
        df_ave=df_ave,
        vl_id=81,
        title="Landmark Motion Relative to Nipple (VL 81)",
        save_path="output/VL81_nipple_relative",
        use_dts_cmap=True
    )

    # Example 2: Plot for another specific subject (e.g., VL 9)
    print("\n" + "="*70)
    print("Example 2: Plotting nipple-relative motion for Subject VL 9")
    print("="*70)

    plot_nipple_relative_landmarks(
        df_ave=df_ave,
        vl_id=81,
        title="Landmark Motion Relative to Nipple (VL 81)",
        save_path="output/VL9_nipple_relative",
        use_dts_cmap=True
    )

    # Example 3: Plot for all subjects combined
    print("\n" + "="*70)
    print("Example 3: Plotting nipple-relative motion for ALL subjects")
    print("="*70)

    plot_nipple_relative_landmarks(
        df_ave=df_ave,
        vl_id=None,  # None means use all subjects
        title="Landmark Motion Relative to Nipple (All Subjects)",
        save_path="output/all_subjects_nipple_relative",
        use_dts_cmap=True
    )

    print("\n" + "="*70)
    print("✓ All plots generated successfully!")
    print("="*70)
    print("\nThe function now reads nipple coordinates directly from DataFrame columns:")
    print("  - left nipple prone transformed x/y/z")
    print("  - left nipple supine x/y/z")
    print("  - right nipple prone transformed x/y/z")
    print("  - right nipple supine x/y/z")
    print("\nNo manual extraction needed!")

if __name__ == "__main__":
    main()
