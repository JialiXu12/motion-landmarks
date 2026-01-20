"""Test script for 3-panel displacement mechanism figure"""
import sys
import traceback

try:
    from analysis import read_data, plot_3panel_displacement_mechanism, EXCEL_FILE_PATH
    import matplotlib
    matplotlib.use('Agg')

    print("Loading data...")
    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)
    print(f"Loaded {len(df_ave)} landmarks")

    print("\nGenerating 3-panel figure...")
    result = plot_3panel_displacement_mechanism(df_ave)

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Nipple displacement:    {result['nipple_displacement']:.1f} mm")
    print(f"Landmark displacement:  {result['landmark_displacement']:.1f} mm")
    print(f"Relative displacement:  {result['relative_displacement']:.1f} mm")
    print(f"Nipple X:               {result['nipple_dx']:.1f} mm")
    print(f"Landmark X:             {result['landmark_dx']:.1f} mm")
    print(f"Relative X (MEDIAL):    {result['relative_dx']:.1f} mm")
    print("="*60)
    print("\n✅ SUCCESS!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
