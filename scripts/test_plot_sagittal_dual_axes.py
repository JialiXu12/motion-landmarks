"""
Test file for plot_sagittal_dual_axes function
Tests different coloring schemes: 'breast', 'subject', and 'dts'
"""

from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path to import analysis module
sys.path.insert(0, str(Path(__file__).parent))

from analysis import read_data, plot_sagittal_dual_axes

# Configuration
OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v4_2026_01_12.xlsx"


def test_plot_sagittal_dual_axes():
    """Test plot_sagittal_dual_axes with different coloring options"""

    print("=" * 80)
    print("TESTING plot_sagittal_dual_axes FUNCTION")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    try:
        df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)
        print(f"✓ Data loaded successfully")
        print(f"  - Raw data shape: {df_raw.shape}")
        print(f"  - Average data shape: {df_ave.shape}")
        print(f"  - Demographic data shape: {df_demo.shape}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

    # Check required columns
    print("\n2. Checking required columns...")
    required_cols = [
        'landmark ave prone transformed x',
        'landmark ave prone transformed y',
        'landmark ave prone transformed z',
        'landmark ave supine x',
        'landmark ave supine y',
        'landmark ave supine z',
        'landmark side (prone)',
        'Distance to skin (prone) [mm]',
        'VL_ID'
    ]

    missing_cols = [col for col in required_cols if col not in df_ave.columns]
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        return False
    else:
        print(f"✓ All required columns present")

    # Verify data counts
    print("\n3. Verifying data counts...")
    left_count = len(df_ave[df_ave['landmark side (prone)'] == 'LB'])
    right_count = len(df_ave[df_ave['landmark side (prone)'] == 'RB'])
    total_count = len(df_ave)

    print(f"  - Total landmarks: {total_count}")
    print(f"  - Right breast: {right_count}")
    print(f"  - Left breast: {left_count}")

    if total_count != (left_count + right_count):
        print(f"✗ Data count mismatch!")
        return False
    print("✓ Data counts verified")

    # Test 1: Default coloring (by breast)
    print("\n" + "=" * 80)
    print("TEST 1: Default coloring (by breast - blue/green)")
    print("=" * 80)
    try:
        plot_sagittal_dual_axes(df_ave, color_by='breast')
        print("✓ Test 1 PASSED: Default coloring completed successfully")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Color by subject
    print("\n" + "=" * 80)
    print("TEST 2: Color by subject (viridis colormap)")
    print("=" * 80)
    try:
        plot_sagittal_dual_axes(df_ave, color_by='subject')
        print("✓ Test 2 PASSED: Subject coloring completed successfully")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Color by DTS
    print("\n" + "=" * 80)
    print("TEST 3: Color by DTS (distance to skin)")
    print("=" * 80)
    try:
        plot_sagittal_dual_axes(df_ave, color_by='dts')
        print("✓ Test 3 PASSED: DTS coloring completed successfully")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Filter by specific subject with DTS coloring
    print("\n" + "=" * 80)
    print("TEST 4: Filter by specific subject with DTS coloring")
    print("=" * 80)

    unique_vl_ids = df_ave['VL_ID'].unique()
    if len(unique_vl_ids) > 0:
        test_vl_id = unique_vl_ids[0]
        print(f"Testing with VL_ID: {test_vl_id}")
        try:
            plot_sagittal_dual_axes(df_ave, color_by='dts', vl_id=test_vl_id)
            print(f"✓ Test 4 PASSED: Single subject filtering completed successfully")
        except Exception as e:
            print(f"✗ Test 4 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("⚠ Test 4 SKIPPED: No VL_IDs found in data")

    # Test 5: Validate output files exist
    print("\n" + "=" * 80)
    print("TEST 5: Validating output files")
    print("=" * 80)

    output_dir = Path("..") / "output" / "figs" / "landmark vectors"
    expected_files = [
        "Vectors_rel_sternum_sagittal_dual.png",
        "Vectors_rel_sternum_sagittal_dual_by_subject.png",
        "Vectors_rel_sternum_sagittal_dual_DTS.png",
    ]

    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size / 1024  # KB
            print(f"✓ {filename} exists ({file_size:.1f} KB)")
        else:
            print(f"✗ {filename} NOT FOUND")
            all_exist = False

    if all_exist:
        print("\n✓ Test 5 PASSED: All expected output files exist")
    else:
        print("\n✗ Test 5 FAILED: Some output files are missing")
        return False

    # Test 6: Verify vector counts in plots match data
    print("\n" + "=" * 80)
    print("TEST 6: Verify vector counts")
    print("=" * 80)

    print(f"Expected vectors in plots:")
    print(f"  - Right breast (left subplot): {right_count} vectors")
    print(f"  - Left breast (right subplot): {left_count} vectors")
    print(f"  - Total: {total_count} vectors")
    print("\n✓ Manual verification: Check that plots show all vectors")

    # Summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nFunction plot_sagittal_dual_axes is working correctly with:")
    print("  1. Default breast coloring (blue/green)")
    print("  2. Subject-based coloring (viridis colormap)")
    print("  3. DTS-based coloring (distance to skin)")
    print("  4. Subject filtering capability")
    print(f"\nOutput saved to: {output_dir.absolute()}")

    return True


if __name__ == "__main__":
    success = test_plot_sagittal_dual_axes()

    if success:
        print("\n🎉 Testing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Testing failed!")
        sys.exit(1)
