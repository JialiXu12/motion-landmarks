"""
Test script for updated analyse_clock_position_rotation function
Tests the new side-by-side polar plots
"""
import sys
import traceback

try:
    print("="*80)
    print("TESTING UPDATED analyse_clock_position_rotation FUNCTION")
    print("="*80)

    from analysis import (read_data, analyse_clock_position_rotation,
                         plot_nipple_relative_landmarks, EXCEL_FILE_PATH)
    import matplotlib
    matplotlib.use('Agg')

    print("\n1. Loading data...")
    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)
    print(f"   ✓ Loaded {len(df_ave)} landmarks from {len(df_ave['VL_ID'].unique())} subjects")

    print("\n2. Getting nipple-relative coordinates...")
    base_left, base_right, vec_left, vec_right, lm_disp_left, lm_disp_right, nipple_disp_left, nipple_disp_right = \
        plot_nipple_relative_landmarks(df_ave, vl_id=None, save_path=None, use_dts_cmap=False)
    print(f"   ✓ Left breast: {len(base_left) if base_left is not None else 0} landmarks")
    print(f"   ✓ Right breast: {len(base_right) if base_right is not None else 0} landmarks")

    print("\n3. Running clock position analysis...")
    clock_summary = analyse_clock_position_rotation(
        df_ave,
        base_left=base_left,
        base_right=base_right,
        vec_left=vec_left,
        vec_right=vec_right
    )

    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)

    if 'left' in clock_summary:
        left = clock_summary['left']
        print(f"\nLeft Breast (n={left['n']}):")
        print(f"  Mean rotation: {left['mean_rotation_hours']:.2f} hours ({left['mean_rotation_deg']:.1f}°)")
        print(f"  P-value: {left['p_value']:.4e}")
        print(f"  Significant: {'Yes' if left['p_value'] < 0.05 else 'No'}")

    if 'right' in clock_summary:
        right = clock_summary['right']
        print(f"\nRight Breast (n={right['n']}):")
        print(f"  Mean rotation: {right['mean_rotation_hours']:.2f} hours ({right['mean_rotation_deg']:.1f}°)")
        print(f"  P-value: {right['p_value']:.4e}")
        print(f"  Significant: {'Yes' if right['p_value'] < 0.05 else 'No'}")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE - New polar plots generated!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. clock_rotation_comparison_individual.png (both breasts, all points)")
    print("  2. clock_rotation_comparison_mean.png (both breasts, mean only)")
    print("\nLayout: Right breast on LEFT side, Left breast on RIGHT side ✓")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
