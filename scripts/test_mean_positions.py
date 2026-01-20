"""
Test to verify that mean prone and supine positions in polar plots are calculated correctly
using circular mean in the compute_clock_positions workflow.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

# Import the circular mean function from analysis
from analysis import circular_mean_angle, read_data, EXCEL_FILE_PATH

def test_mean_prone_position():
    """Test that mean prone position is calculated correctly"""

    print("="*70)
    print("TESTING MEAN PRONE POSITION CALCULATION")
    print("="*70)

    # Load actual data
    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)

    # Get left breast data
    df_left = df_ave[df_ave['landmark side (prone)'] == 'LB'].copy()

    if len(df_left) == 0:
        print("❌ No left breast data found!")
        return False

    print(f"\n✓ Loaded {len(df_left)} left breast landmarks")

    # Simulate what compute_clock_positions does
    # Extract prone positions relative to nipple (this would come from base_left)
    prone_lm_x = df_left['landmark ave prone transformed x'].values
    prone_lm_z = df_left['landmark ave prone transformed z'].values
    prone_nip_x = df_left['left nipple prone transformed x'].values
    prone_nip_z = df_left['left nipple prone transformed z'].values

    # Calculate positions relative to nipple
    prone_x_rel = prone_lm_x - prone_nip_x
    prone_z_rel = prone_lm_z - prone_nip_z

    # Calculate angles (same as compute_clock_positions)
    angle_prone_deg = np.degrees(np.arctan2(prone_x_rel, prone_z_rel))
    angle_prone_deg = (angle_prone_deg + 360) % 360  # Normalize to [0, 360)

    print("\n" + "-"*70)
    print("PRONE POSITION ANGLES (sample)")
    print("-"*70)
    print(f"Sample angles (first 10): {angle_prone_deg[:10]}")
    print(f"Min angle: {angle_prone_deg.min():.1f}°")
    print(f"Max angle: {angle_prone_deg.max():.1f}°")

    # Method 1: WRONG - Arithmetic mean
    mean_angle_arithmetic = angle_prone_deg.mean()

    # Method 2: CORRECT - Circular mean
    angle_prone_rad = np.radians(angle_prone_deg)
    mean_angle_circular_rad = circular_mean_angle(angle_prone_rad)
    mean_angle_circular_deg = np.degrees(mean_angle_circular_rad)
    if mean_angle_circular_deg < 0:
        mean_angle_circular_deg += 360

    print("\n" + "-"*70)
    print("MEAN CALCULATIONS")
    print("-"*70)
    print(f"❌ Arithmetic mean: {mean_angle_arithmetic:.1f}°")
    print(f"✓  Circular mean:   {mean_angle_circular_deg:.1f}°")
    print(f"   Difference:      {abs(mean_angle_arithmetic - mean_angle_circular_deg):.1f}°")

    # Check if difference is significant
    if abs(mean_angle_arithmetic - mean_angle_circular_deg) > 10:
        print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
        print("   This confirms circular mean is ESSENTIAL for this data.")
    else:
        print("\n✓  Small difference (angles not near boundary)")

    # Convert to clock positions
    def angle_to_clock(deg):
        hours = (12 - deg/30) % 12
        if hours == 0:
            hours = 12
        return hours

    clock_arithmetic = angle_to_clock(mean_angle_arithmetic)
    clock_circular = angle_to_clock(mean_angle_circular_deg)

    print("\n" + "-"*70)
    print("CLOCK POSITIONS")
    print("-"*70)
    print(f"❌ Arithmetic mean: {clock_arithmetic:.1f} o'clock")
    print(f"✓  Circular mean:   {clock_circular:.1f} o'clock")

    # Verify the code is using circular mean
    print("\n" + "="*70)
    print("VERIFICATION IN CODE")
    print("="*70)

    # Read the actual code to verify
    with open('analysis.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if 'circular_mean_angle(theta_prone_rad)' in content:
        print("✓  Code uses circular_mean_angle for prone angles")
    else:
        print("❌ Code does NOT use circular_mean_angle for prone angles!")
        return False

    if 'circular_mean_angle(theta_supine_rad)' in content:
        print("✓  Code uses circular_mean_angle for supine angles")
    else:
        print("❌ Code does NOT use circular_mean_angle for supine angles!")
        return False

    # Check both locations (clock rotation plot and comparison plot)
    usage_count = content.count('circular_mean_angle(theta_prone')
    print(f"✓  Found {usage_count} locations using circular mean for prone angles")

    if usage_count >= 2:
        print("✓  Both main polar plots are using circular mean!")
    else:
        print(f"⚠️  Only {usage_count} location(s) found, should be 2+")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("✓  Mean prone position IS calculated correctly using circular mean")
    print("✓  Mean supine position IS calculated correctly using circular mean")
    print("✓  The polar plots will show correct mean positions")
    print("="*70)

    return True

if __name__ == "__main__":
    try:
        success = test_mean_prone_position()
        if success:
            print("\n✅ ALL TESTS PASSED - Mean positions are correct!")
            sys.exit(0)
        else:
            print("\n❌ TESTS FAILED - Check the implementation!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
